from __future__ import annotations

from slanq.ast_nodes import (
    Block,
    Call,
    Declaration,
    Expression,
    For,
    Index,
    Name,
    ProcessDef,
    Program,
    QuantumDecl,
    Span,
    Symbol,
    qubit_count,
)
from slanq.builtin import BUILTIN_NAMES, RANGE, ConstEvalError, const_int, const_value
from slanq.diagnostics import DiagnosticBag
from slanq.visitor import NodeVisitor

Scope = dict[str, Symbol]


def analyze(ast: Program, bag: DiagnosticBag) -> None:
    scope = _build_symbol_table(ast, bag)
    _resolve_names(ast, bag, scope)
    _check_types(ast, bag)
    _check_affine(ast, bag)


def _build_symbol_table(ast: Program, bag: DiagnosticBag) -> Scope:
    """Collect the top-level declarations so they can be referenced before their
    definition; nested scopes are populated during name resolution instead."""
    scope: Scope = {}
    for statement in ast.statements:
        if not isinstance(statement, Declaration):
            continue
        if statement.name in scope:
            _error(bag, f"duplicate declaration of '{statement.name}'", statement.span)
            continue
        scope[statement.name] = statement
    return scope


class _NameResolver(NodeVisitor):
    def __init__(self, global_scope: Scope, bag: DiagnosticBag) -> None:
        self.scopes: list[Scope] = [global_scope]
        self.bag = bag

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, Declaration):
            self._declare(node)

    def visit_Block(self, node: Block) -> None:
        self.scopes.append({})
        for statement in node.statements:
            self.visit(statement)
        self.scopes.pop()

    def visit_For(self, node: For) -> None:
        # The loop variable is not yet in scope in the range expression itself.
        self.visit(node.iterable)
        self.scopes.append({})
        self._declare(node.binding)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ProcessDef(self, node: ProcessDef) -> None:
        self._declare(node)
        self.scopes.append({})
        for parameter in node.params:
            self._declare(parameter)
        self.visit(node.body)
        self.scopes.pop()

    def visit_Name(self, node: Name) -> None:
        if node.name in BUILTIN_NAMES:
            return
        symbol = self._lookup(node.name)
        if symbol is None:
            _error(self.bag, f"undefined name '{node.name}'", node.span)
        else:
            node.resolved_symbol = symbol

    def _declare(self, symbol: Symbol) -> None:
        current = self.scopes[-1]
        existing = current.get(symbol.name)
        if existing is symbol:
            return
        if existing is not None:
            _error(self.bag, f"duplicate declaration of '{symbol.name}'", symbol.span)
            return
        current[symbol.name] = symbol

    def _lookup(self, name: str) -> Symbol | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None


def _resolve_names(ast: Program, bag: DiagnosticBag, scope: Scope) -> None:
    _NameResolver(scope, bag).visit(ast)


class _Checker(NodeVisitor):
    """Shared plumbing for the checks that evaluate constant subexpressions."""

    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag

    def _value(self, expression: Expression) -> int | float | bool | None:
        try:
            return const_value(expression)
        except ConstEvalError as exc:
            self._reject(str(exc), expression)
            return None

    def _int(self, expression: Expression) -> int | None:
        try:
            return const_int(expression)
        except ConstEvalError as exc:
            self._reject(str(exc), expression)
            return None

    def _reject(self, message: str, node: Expression) -> None:
        _error(self.bag, message, node.span)


class _IndexChecker(_Checker):
    def visit_For(self, node: For) -> None:
        # The body is skipped: the loop itself is not lowered yet, so a loop
        # variable has no compile-time value and every index in the body would
        # be reported as non-constant. That message would hide the real one.
        self.visit(node.iterable)

    def visit_Index(self, node: Index) -> None:
        self.generic_visit(node)

        size = _quantum_size_of(node.base)
        if size is None:
            # Not a quantum register; classical arrays (`gamma[i]`) come later.
            return

        index = self._int(node.index)
        if index is None:
            self._reject(
                "a quantum register index must be a compile-time integer", node.index
            )
            return

        if not 0 <= index < size:
            self._reject(
                f"index {index} is out of range for '{node.base.name}' of size {size}",
                node.index,
            )

    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        size = qubit_count(node.declared_type)
        if size is None or not isinstance(node.initializer, Expression):
            return

        value = self._value(node.initializer)
        if not isinstance(value, int):
            return

        largest = 2**size - 1
        if not 0 <= value <= largest:
            self._reject(
                f"{value} does not fit in '{node.name}', which holds {size} "
                f"qubit(s) (allowed: 0..{largest})",
                node.initializer,
            )


class _ForChecker(_Checker):
    def visit_For(self, node: For) -> None:
        self.generic_visit(node)

        iterable = node.iterable
        if not (isinstance(iterable, Call) and iterable.callee.name == RANGE):
            self._reject(f"a for loop iterates over {RANGE}(...)", iterable)
            return

        if not 1 <= len(iterable.args) <= 3:
            self._reject(
                f"{RANGE}() takes a stop, a start and a stop, or a start, a stop "
                f"and a step -- got {len(iterable.args)} argument(s)",
                iterable,
            )
            return

        if len(iterable.args) == 3 and self._value(iterable.args[2]) == 0:
            self._reject("the step of a for loop must not be zero", iterable.args[2])


def _quantum_size_of(base: Name) -> int | None:
    symbol = base.resolved_symbol
    if not isinstance(symbol, QuantumDecl):
        return None
    return qubit_count(symbol.declared_type)


def _check_types(ast: Program, bag: DiagnosticBag) -> None:
    _IndexChecker(bag).visit(ast)
    _ForChecker(bag).visit(ast)


def _check_affine(ast: Program, bag: DiagnosticBag) -> None:
    pass


def _error(bag: DiagnosticBag, message: str, span: Span) -> None:
    bag.error(message, line=span.start_line, column=span.start_col)


__all__ = ["analyze"]
