"""AST-to-AST transformations that run between name resolution and type
checking, so every later phase sees one fully expanded tree."""

from __future__ import annotations

import dataclasses
from typing import Any

from slanq.ast_nodes import (
    Block,
    Call,
    Declaration,
    Expression,
    ExprStatement,
    Index,
    Name,
    Node,
    ProcessDef,
    Program,
    Span,
    Statement,
)
from slanq.diagnostics import DiagnosticBag

# A body is copied into every call site, so a chain where each process calls
# the next one twice doubles the statement count per level. Bounded so a short
# program cannot hang the compiler -- the same class of guard const_value's
# `**` carries.
MAX_EXPANDED_STATEMENTS = 10_000


def expand_processes(ast: Program, bag: DiagnosticBag) -> None:
    """Replace every call to a `process` with a copy of its body, the
    parameters substituted by that call's arguments.

    Runs after name resolution: a copied name that is not a parameter keeps
    the symbol it already resolved to, so no second resolution pass is
    needed. Every later check therefore sees the substituted shape and
    validates it against the actual arguments for free -- index bounds
    against the real register size, `a += a` self-reference, qif overlap.
    """
    expander = _Expander(bag)
    # Every definition is checked before any call is expanded: a call may
    # precede the definition it refers to, since top-level names are collected
    # ahead of resolution.
    for statement in ast.statements:
        if isinstance(statement, ProcessDef):
            expander.check_definition(statement)
    ast.statements = expander.rewrite(ast.statements, top_level=True)


class _Expander:
    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        # Process names currently being expanded, so a cycle is caught as one
        # is re-entered; this covers mutual recursion, not just self-calls.
        self.active: list[str] = []
        self.budget = MAX_EXPANDED_STATEMENTS
        self.reported_budget = False
        # Set while copying a body if substitution produced an unreadable
        # shape, so that expansion is abandoned instead of half-applied.
        self.invalid = False
        # Processes already reported as impossible to expand; their calls
        # expand to nothing rather than to a body known to be broken.
        self.unexpandable: set[str] = set()

    def rewrite(self, statements: list[Statement], *, top_level: bool) -> list[Statement]:
        result: list[Statement] = []
        for statement in statements:
            if isinstance(statement, ProcessDef):
                if top_level:
                    # A definition is a template: it stays in the tree so its
                    # body is still checked, and lowers to nothing.
                    result.append(statement)
                else:
                    self._error(
                        "a process may only be defined at the top level of a "
                        "program",
                        statement.span,
                    )
                continue

            for block in _child_blocks(statement):
                block.statements = self.rewrite(block.statements, top_level=False)

            expansion = self._expand_call(statement)
            result.extend([statement] if expansion is None else expansion)
        return result

    def check_definition(self, process: ProcessDef) -> None:
        """A declaration in a body would be copied once per call, and two
        copies claim the same register name -- the generated file then fails
        to build (`register name "t" already exists`). Reported once, at the
        definition, rather than once per call site; the process is then
        recorded as unexpandable so its call sites add nothing but silence,
        instead of follow-on errors about a name that was left behind.

        Only the body's own statements are scanned: a declaration deeper than
        that sits inside an `if`/`for`/`qif`, each of which is already
        reported on its own.
        """
        for statement in process.body.statements:
            if isinstance(statement, Declaration):
                self._error(
                    "a declaration inside a process body is not implemented "
                    "yet; this is a limitation of the compiler, not an error "
                    "in the program",
                    statement.span,
                )
                self.unexpandable.add(process.name)

    def _expand_call(self, statement: Statement) -> list[Statement] | None:
        """The statements a process call expands to, or None if `statement` is
        not a process call at all."""
        if not isinstance(statement, ExprStatement) or not isinstance(statement.expr, Call):
            return None
        call = statement.expr
        process = call.callee.resolved_symbol
        if not isinstance(process, ProcessDef):
            return None

        if process.name in self.unexpandable:
            return []

        if len(call.args) != len(process.params):
            # Analysis reports the arity itself, while the call is still in
            # the tree; dropping it keeps an unexpandable call out of lowering.
            return []

        if process.name in self.active:
            self._error(
                f"'{process.name}' calls itself, directly or through another "
                "process; a process is expanded at its call site, so it cannot "
                "be recursive",
                call.span,
            )
            return []

        bindings = {
            id(parameter): argument
            for parameter, argument in zip(process.params, call.args, strict=True)
        }

        self.invalid = False
        body: list[Statement] = []
        for source in process.body.statements:
            copied = self._copy(source, bindings)
            assert isinstance(copied, Statement)
            body.append(copied)
        if self.invalid:
            return []

        self.budget -= len(body)
        if self.budget < 0:
            if not self.reported_budget:
                self.reported_budget = True
                self._error(
                    f"expanding process calls produced more than "
                    f"{MAX_EXPANDED_STATEMENTS} statements; this is not "
                    "compiled",
                    call.span,
                )
            return []

        self.active.append(process.name)
        expanded = self.rewrite(body, top_level=False)
        self.active.pop()
        return expanded

    def _copy(self, node: Node, bindings: dict[int, Expression]) -> Node:
        """A copy of `node` with every reference to a bound parameter replaced
        by that parameter's argument.

        Fields marked `metadata={"annotation": True}` hold a back-reference to
        a declaration, not tree structure, so they are shared rather than
        copied -- which is exactly what keeps a copied name resolved. Spans are
        shared too, so a diagnostic from inside a body points at the line of
        the process, where the offending code is actually written.
        """
        if isinstance(node, Name):
            argument = bindings.get(id(node.resolved_symbol))
            if argument is not None:
                return self._copy(argument, {})

        values: dict[str, Any] = {}
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            if f.metadata.get("annotation") or not isinstance(value, Node | list):
                continue
            if isinstance(value, Node):
                values[f.name] = self._copy(value, bindings)
            else:
                values[f.name] = [
                    self._copy(item, bindings) if isinstance(item, Node) else item
                    for item in value
                ]
        copied = dataclasses.replace(node, **values)

        # Two fields are typed as a bare Name rather than any expression, so
        # substituting something else there would leave a shape no later phase
        # can read (`num[1][0]`). Rejected instead; `invalid` makes the caller
        # drop the whole expansion, since a partially substituted body would
        # only produce follow-on errors about the parameter name.
        if isinstance(copied, Index) and not isinstance(copied.base, Name):
            self._error(
                "a process parameter that is indexed in the process body must "
                "be given a whole quantum variable, not a single qubit",
                node.span,
            )
            self.invalid = True
        elif isinstance(copied, Call) and not isinstance(copied.callee, Name):
            self._error("a process parameter cannot be called", node.span)
            self.invalid = True
        return copied

    def _error(self, message: str, span: Span) -> None:
        self.bag.error(message, line=span.start_line, column=span.start_col)


def _child_blocks(statement: Statement) -> list[Block]:
    blocks = []
    for f in dataclasses.fields(statement):
        value = getattr(statement, f.name)
        if isinstance(value, Block):
            blocks.append(value)
    return blocks


__all__ = ["expand_processes"]
