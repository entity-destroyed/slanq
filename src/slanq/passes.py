"""AST-to-AST transformations that run between name resolution and type
checking, so every later phase sees one fully expanded tree."""

from __future__ import annotations

import dataclasses
from typing import Any

from slanq.ast_nodes import (
    Assign,
    AugAssign,
    BinaryOp,
    Block,
    Call,
    ClassicalDecl,
    Declaration,
    Expression,
    ExprStatement,
    For,
    Index,
    Literal,
    LoopVarDecl,
    Name,
    Node,
    ParamArrayDecl,
    ParamDecl,
    ProcessDef,
    Program,
    Span,
    Statement,
    UnknownValue,
)
from slanq.builtin import RANGE, ConstEvalError, const_value
from slanq.diagnostics import DiagnosticBag
from slanq.visitor import iter_child_nodes

# A body is copied into every call site and once per iteration, so a chain
# where each process calls the next one twice, or a loop nested in a loop,
# multiplies the statement count per level. Bounded so a short program cannot
# hang the compiler -- the same class of guard const_value's `**` carries.
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


def unroll_loops(ast: Program, bag: DiagnosticBag) -> None:
    """Replace every `for` with its body repeated once per iteration, the loop
    variable substituted by that iteration's value.
    """
    ast.statements = _Unroller(bag).rewrite(ast.statements)


class _Copier:
    """Copies a body, replacing some of the names in it with expressions.

    Fields marked `metadata={"annotation": True}` hold a back-reference to a
    declaration, not tree structure, so they are shared rather than copied --
    which is exactly what keeps a copied name resolved. Declarations are
    shared for the same reason read backwards: every reference to one is
    shared, so a copied declaration would be one nothing points at. Spans are
    shared too, so a diagnostic from inside a copied body points at the line
    where the offending code is actually written.
    """

    # Two fields are typed as a bare Name rather than any expression, so
    # substituting something else there would leave a shape no later phase can
    # read (`num[1][0]`).
    index_base_error: str
    callee_error: str

    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        self.budget = MAX_EXPANDED_STATEMENTS
        self.reported_budget = False
        # Set while copying a body if substitution produced an unreadable
        # shape, so that the copy is abandoned instead of half-applied.
        self.invalid = False

    def _substitute(self, node: Name) -> Expression | None:
        """The expression `node` stands for in this copy, or None if it is not
        one of the substituted names."""
        raise NotImplementedError

    def _copy(self, node: Node, *, substitute: bool = True) -> Node:
        if substitute and isinstance(node, Name):
            replacement = self._substitute(node)
            if replacement is not None:
                return replacement

        values: dict[str, Any] = {}
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            if f.metadata.get("annotation") or not isinstance(value, Node | list):
                continue
            if isinstance(value, Declaration):
                continue
            if isinstance(value, Node):
                values[f.name] = self._copy(value, substitute=substitute)
            else:
                values[f.name] = [
                    self._copy(item, substitute=substitute) if isinstance(item, Node) else item
                    for item in value
                ]
        copied = dataclasses.replace(node, **values)

        # `invalid` makes the caller drop the whole copy, since a partially
        # substituted body would only produce follow-on errors about a name
        # that no longer means anything.
        if isinstance(copied, Index) and not isinstance(copied.base, Name):
            self._error(self.index_base_error, node.span)
            self.invalid = True
        elif isinstance(copied, Call) and not isinstance(copied.callee, Name):
            self._error(self.callee_error, node.span)
            self.invalid = True
        return copied

    def _charge(self, statements: int, what: str, span: Span) -> bool:
        """Whether `statements` more fit in the budget. Reported once: a
        program over the limit is usually over it at every call site."""
        self.budget -= statements
        if self.budget >= 0:
            return True
        if not self.reported_budget:
            self.reported_budget = self._error(
                f"{what} produced more than {MAX_EXPANDED_STATEMENTS} "
                "statements; this is not compiled",
                span,
            )
        return False

    def _error(self, message: str, span: Span) -> bool:
        """Whether the message was taken; a subclass may suppress it."""
        self.bag.error(message, line=span.start_line, column=span.start_col)
        return True


class _Expander(_Copier):
    index_base_error = (
        "a process parameter that is indexed in the process body must be given "
        "a whole quantum variable, not a single qubit"
    )
    callee_error = "a process parameter cannot be called"

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        # Process names currently being expanded, so a cycle is caught as one
        # is re-entered; this covers mutual recursion, not just self-calls.
        self.active: list[str] = []
        # Processes already reported as impossible to expand; their calls
        # expand to nothing rather than to a body known to be broken.
        self.unexpandable: set[str] = set()
        self.bindings: dict[int, Expression] = {}

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

    def _substitute(self, node: Name) -> Expression | None:
        argument = self.bindings.get(id(node.resolved_symbol))
        if argument is None:
            return None
        # Copied, not shared: a parameter used twice would otherwise put one
        # argument node in the tree twice. Substitution is off inside it, so an
        # argument that happens to name another parameter is left alone.
        copied = self._copy(argument, substitute=False)
        assert isinstance(copied, Expression)
        return copied

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

        self.bindings = {
            id(parameter): argument
            for parameter, argument in zip(process.params, call.args, strict=True)
        }

        self.invalid = False
        body: list[Statement] = []
        for source in process.body.statements:
            copied = self._copy(source)
            assert isinstance(copied, Statement)
            body.append(copied)
        if self.invalid:
            return []

        if not self._charge(len(body), "expanding process calls", call.span):
            return []

        self.active.append(process.name)
        expanded = self.rewrite(body, top_level=False)
        self.active.pop()
        return expanded


class _Unroller(_Copier):
    index_base_error = "a loop variable holds a number, so it cannot be indexed"
    callee_error = "a loop variable cannot be called"

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self.loop_var: LoopVarDecl | None = None
        self.value = 0
        # Loops reached only by substitution, so that a triangular loop's
        # empty first pass is not warned about like a written `range(0)`.
        self.copied = 0
        self.silent = False

    def rewrite(self, statements: list[Statement]) -> list[Statement]:
        result: list[Statement] = []
        for statement in statements:
            if isinstance(statement, For):
                # Unrolled before anything inside it is walked, so a nested
                # loop whose range mentions the outer variable (`range(i)`) is
                # already constant by the time it is reached.
                result.extend(self._unroll(statement))
                continue
            previous = self.silent
            self.silent = self.silent or isinstance(statement, ProcessDef)
            for block in _child_blocks(statement):
                block.statements = self.rewrite(block.statements)
            self.silent = previous
            result.append(statement)
        return result

    def _error(self, message: str, span: Span) -> bool:
        """A process definition stays in the tree as a template, and its
        parameters have no values there, so `range(n)` is not constant however
        constant every call makes it. Nothing is reported from inside one: the
        call sites re-report it at the same span, and an uncalled process has
        nothing to report about."""
        if self.silent:
            return False
        return super()._error(message, span)

    def _warn_about_empty_loop(self, node: For, values: range) -> None:
        if self.copied or self.silent:
            return
        if not values:
            message = "this loop runs zero times, so nothing in its body is compiled"
        elif not node.body.statements:
            message = "this loop has an empty body, so it does nothing"
        else:
            return
        self.bag.warning(
            message, line=node.span.start_line, column=node.span.start_col
        )

    def _substitute(self, node: Name) -> Expression | None:
        if node.resolved_symbol is not self.loop_var:
            return None
        # The reference's own span, not the loop header's: a diagnostic about
        # the substituted value points at where the variable is written.
        return Literal(span=node.span, value=self.value)

    def _unroll(self, node: For) -> list[Statement]:
        for statement in node.body.statements:
            if isinstance(statement, Declaration):
                # The same failure as a declaration in a process body: the
                # second iteration claims a register name the first one took.
                self._error(
                    "a declaration inside a loop body is not implemented yet; "
                    "this is a limitation of the compiler, not an error in the "
                    "program",
                    statement.span,
                )
                return []

        values = self._iteration_values(node)
        if values is None:
            return []
        self._warn_about_empty_loop(node, values)
        if len(values) == 0:
            return []

        # Charged before anything is copied: an iteration count large enough to
        # exhaust memory must not be reached one statement at a time. An empty
        # body still costs one per iteration, so it cannot spin for free.
        cost = len(values) * max(1, len(node.body.statements))
        if not self._charge(cost, "unrolling a for loop", node.span):
            return []

        body: list[Statement] = []
        for value in values:
            self.invalid = False
            iteration = self._copy_iteration(node, value)
            if self.invalid:
                return []
            body.extend(iteration)

        self.copied += 1
        expanded = self.rewrite(body)
        self.copied -= 1
        return expanded

    def _iteration_values(self, node: For) -> range | None:
        """The values the loop variable takes. A range rather than a list, so a
        count too large to unroll can be measured without being built."""
        iterable = node.iterable
        if not (isinstance(iterable, Call) and iterable.callee.name == RANGE):
            return None  # _ForChecker reports the iterable's shape
        if not 1 <= len(iterable.args) <= 3:
            return None  # and its argument count

        bounds: list[int] = []
        for argument in iterable.args:
            value = self._bound(argument)
            if value is None:
                return None
            bounds.append(value)

        step = bounds[2] if len(bounds) == 3 else 1
        if step == 0:
            self._error(
                "the step of a for loop must not be zero", iterable.args[2].span
            )
            return None
        start, stop = (bounds[0], bounds[1]) if len(bounds) > 1 else (0, bounds[0])
        return range(start, stop, step)

    def _bound(self, argument: Expression) -> int | None:
        """One `range` argument as a number. A circuit is built once, with a
        fixed structure, so the count has to be known then; this is not a
        limitation waiting to be lifted, since Qiskit's own `for_loop` takes a
        concrete range too and rejects a `Parameter` as its index set."""
        try:
            value = const_value(argument)
        except ConstEvalError as exc:
            self._error(str(exc), argument.span)
            return None
        if value is None:
            detail = ""
            if isinstance(argument, Name) and isinstance(
                argument.resolved_symbol, ParamDecl | ParamArrayDecl
            ):
                detail = f"; '{argument.name}' only gets its value at runtime"
            elif isinstance(argument, Name) and isinstance(
                argument.effective_value, UnknownValue
            ):
                detail = f"; '{argument.name}': {argument.effective_value.reason}"
            self._error(
                "a for loop needs an iteration count known when the circuit is "
                f"built{detail}",
                argument.span,
            )
            return None
        if type(value) is not int:
            self._error(
                f"a for loop counts in whole numbers, so {RANGE}() cannot take "
                f"{value!r}",
                argument.span,
            )
            return None
        return value

    def _copy_iteration(self, node: For, value: int) -> list[Statement]:
        self.loop_var = node.binding
        self.value = value
        copied: list[Statement] = []
        for source in node.body.statements:
            item = self._copy(source)
            assert isinstance(item, Statement)
            copied.append(item)
        return copied


BRANCH_MERGE = "its value here depends on a branch"
BEFORE_DECLARATION = "it has no value before its declaration"


def track_values(ast: Program) -> None:
    """Annotate every name with the expression in effect where it stands.

    Without assignment a name always means its initializer, which is why
    `const_value` could read that directly. With assignment it means whatever
    was last written to it, and after two branches meet it means nothing the
    compiler can name -- an honest answer the places needing a compile-time
    value can report."""
    _Tracker().statements(ast.statements, {})


def _collect_assigned(
    statements: list[Statement], found: dict[int, ClassicalDecl]
) -> None:
    for statement in statements:
        if isinstance(statement, Assign | AugAssign) and isinstance(
            statement.target, Name
        ):
            symbol = statement.target.resolved_symbol
            if isinstance(symbol, ClassicalDecl):
                found[id(symbol)] = symbol
        for block in _child_blocks(statement):
            _collect_assigned(block.statements, found)


# Keyed by identity: an AST node carries a dataclass __eq__, so two
# declarations that happen to look alike would collide.
Environment = dict[int, "Expression | UnknownValue"]


class _Tracker:
    def statements(self, statements: list[Statement], env: Environment) -> None:
        for statement in statements:
            self._statement(statement, env)

    def _statement(self, statement: Statement, env: Environment) -> None:
        blocks = _child_blocks(statement)
        inner: dict[int, ClassicalDecl] = {}
        for block in blocks:
            _collect_assigned(block.statements, inner)

        for child in iter_child_nodes(statement):
            if not isinstance(child, Block):
                self._annotate(child, env)

        if isinstance(statement, ClassicalDecl):
            env[id(statement)] = statement.initializer
            return

        if isinstance(statement, Assign | AugAssign) and isinstance(
            statement.target, Name
        ):
            symbol = statement.target.resolved_symbol
            if isinstance(symbol, ClassicalDecl):
                env[id(symbol)] = _written_value(statement, env.get(id(symbol)))
                return

        if not blocks:
            return

        # Which branch runs is not settled here, so a name either arm writes
        # has no one value afterwards.
        for block in blocks:
            self.statements(block.statements, dict(env))
        for key in inner:
            env[key] = UnknownValue(reason=BRANCH_MERGE)

    def _annotate(self, node: Node, env: Environment) -> None:
        if isinstance(node, Name):
            symbol = node.resolved_symbol
            if isinstance(symbol, ClassicalDecl):
                node.effective_value = env.get(
                    id(symbol), UnknownValue(reason=BEFORE_DECLARATION)
                )
        for child in iter_child_nodes(node):
            self._annotate(child, env)


def _written_value(
    statement: Assign | AugAssign, current: Expression | UnknownValue | None
) -> Expression | UnknownValue:
    if isinstance(statement, Assign):
        return statement.value
    if current is None or isinstance(current, UnknownValue):
        return UnknownValue(reason=BRANCH_MERGE)
    return BinaryOp(
        span=statement.span,
        op=statement.op.removesuffix("="),
        left=current,
        right=statement.value,
    )


def _child_blocks(statement: Statement) -> list[Block]:
    blocks = []
    for f in dataclasses.fields(statement):
        value = getattr(statement, f.name)
        if isinstance(value, Block):
            blocks.append(value)
    return blocks


__all__ = ["expand_processes", "track_values", "unroll_loops"]
