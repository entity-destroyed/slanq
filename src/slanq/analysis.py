from __future__ import annotations

from slanq.ast_nodes import (
    Block,
    BuiltinDecl,
    Call,
    Declaration,
    Expression,
    ExprStatement,
    For,
    Index,
    Name,
    ProcessDef,
    ProcParam,
    Program,
    QuantumDecl,
    Span,
    Symbol,
    Type,
    is_quantum,
    qubit_count,
)
from slanq.builtin import (
    BUILTIN_SCOPE,
    RANGE,
    ArgKind,
    ConstEvalError,
    const_int,
    const_value,
)
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
        if statement.name in BUILTIN_SCOPE:
            _error(bag, _shadowing_message(statement.name), statement.span)
            continue
        if statement.name in scope:
            _error(bag, f"duplicate declaration of '{statement.name}'", statement.span)
            continue
        scope[statement.name] = statement
    return scope


def _shadowing_message(name: str) -> str:
    return f"'{name}' is built into the language and cannot be redeclared"


class _NameResolver(NodeVisitor):
    def __init__(self, global_scope: Scope, bag: DiagnosticBag) -> None:
        self.scopes: list[Scope] = [dict(BUILTIN_SCOPE), global_scope]
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
        if symbol.name in BUILTIN_SCOPE:
            _error(self.bag, _shadowing_message(symbol.name), symbol.span)
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


class _CallChecker(_Checker):
    """Arity, argument kinds and broadcast shape for calls to builtins."""

    def __init__(self, bag: DiagnosticBag) -> None:
        super().__init__(bag)
        self.statement_call: Expression | None = None

    def visit_ExprStatement(self, node: ExprStatement) -> None:
        previous = self.statement_call
        self.statement_call = node.expr
        self.generic_visit(node)
        self.statement_call = previous

    def visit_For(self, node: For) -> None:
        # `range(...)` is checked by _ForChecker, but its arguments are ordinary
        # expressions and may contain calls of their own.
        iterable = node.iterable
        if isinstance(iterable, Call) and iterable.callee.name == RANGE:
            for argument in iterable.args:
                self.visit(argument)
        else:
            self.visit(iterable)
        self.visit(node.body)

    def visit_Call(self, node: Call) -> None:
        self.generic_visit(node)

        symbol = node.callee.resolved_symbol
        if not isinstance(symbol, BuiltinDecl):
            # A call to a `process`; the lowering reports it as unimplemented.
            return

        name = node.callee.name
        signature = symbol.signature
        if signature is None:
            self._reject(f"'{name}' is only valid as the iterable of a for loop", node)
            return

        expected = len(signature.args)
        if len(node.args) != expected:
            self._reject(
                f"'{name}' takes {expected} argument(s), got {len(node.args)}", node
            )
            return

        widths = [
            width
            for kind, argument in zip(signature.args, node.args, strict=True)
            if (width := self._check_argument(name, kind, argument)) is not None
            and kind is ArgKind.QUBITS
        ]
        # Qiskit's rule, measured: equal sizes pair up and a single bit spreads
        # over a register, but two registers of different sizes have no pairing.
        registers = {width for width in widths if width > 1}
        if len(registers) > 1:
            sizes = ", ".join(str(width) for width in sorted(registers))
            self._reject(f"'{name}' cannot combine operands of sizes {sizes}", node)

        node.inferred_type = signature.returns
        self._check_position(name, signature, node)

    def _check_argument(
        self, name: str, kind: ArgKind, argument: Expression
    ) -> int | None:
        operand = _quantum_operand(argument)

        if kind is ArgKind.ANGLE:
            if operand is not None:
                self._reject(f"'{name}' expects a number here, not a qubit", argument)
            return None

        if operand is None:
            self._reject(f"'{name}' expects a quantum variable here", argument)
            return None

        declared, indexed = operand
        if kind is ArgKind.QVAR and indexed:
            self._reject(
                f"'{name}' takes a whole quantum variable, not a single qubit",
                argument,
            )
            return None

        return 1 if indexed else qubit_count(declared)

    def _check_position(self, name: str, signature, node: Call) -> None:
        in_statement = node is self.statement_call
        if in_statement and signature.returns is not None:
            self._reject(f"the result of '{name}' must be assigned", node)
        elif not in_statement and signature.returns is None:
            self._reject(f"'{name}' does not return a value", node)


def _quantum_operand(expression: Expression) -> tuple[Type, bool] | None:
    """The declared type of a qubit operand and whether it is indexed."""
    if isinstance(expression, Index):
        declared = _declared_type(expression.base)
        return (declared, True) if declared is not None else None
    if isinstance(expression, Name):
        declared = _declared_type(expression)
        return (declared, False) if declared is not None else None
    return None


def _declared_type(name: Name) -> Type | None:
    symbol = name.resolved_symbol
    if not isinstance(symbol, QuantumDecl | ProcParam):
        return None
    return symbol.declared_type if is_quantum(symbol.declared_type) else None


def _quantum_size_of(base: Name) -> int | None:
    symbol = base.resolved_symbol
    if not isinstance(symbol, QuantumDecl):
        return None
    return qubit_count(symbol.declared_type)


def _check_types(ast: Program, bag: DiagnosticBag) -> None:
    _IndexChecker(bag).visit(ast)
    _ForChecker(bag).visit(ast)
    _CallChecker(bag).visit(ast)


def _check_affine(ast: Program, bag: DiagnosticBag) -> None:
    pass


def _error(bag: DiagnosticBag, message: str, span: Span) -> None:
    bag.error(message, line=span.start_line, column=span.start_col)


__all__ = ["analyze"]
