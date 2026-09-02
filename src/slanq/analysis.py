from __future__ import annotations

from slanq.ast_nodes import (
    AugAssign,
    BinaryOp,
    Block,
    BuiltinDecl,
    Call,
    Declaration,
    Expression,
    ExprStatement,
    For,
    Index,
    Name,
    Node,
    ParamArrayDecl,
    ParamDecl,
    ProbList,
    ProcessDef,
    ProcParam,
    Program,
    QIf,
    QuantumDecl,
    Span,
    Symbol,
    Type,
    UnaryOp,
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
from slanq.visitor import NodeVisitor, iter_child_nodes

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

    def _reject(self, message: str, node: Node) -> None:
        _error(self.bag, message, node.span)

    def _warn(self, message: str, node: Node) -> None:
        self.bag.warning(message, line=node.span.start_line, column=node.span.start_col)


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


# Below this, a probability list is treated as "meant to sum to 1": floating
# point noise from ordinary decimal literals is silently normalized away rather than warned about.
PROBABILITY_SUM_TOLERANCE = 1e-9


class _ProbListChecker(_Checker):
    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        if not isinstance(node.initializer, ProbList):
            return

        probabilities = node.initializer.probabilities
        if not probabilities:
            # An empty list means equal superposition; nothing to validate.
            return

        size = qubit_count(node.declared_type)
        expected = 2**size
        if len(probabilities) != expected:
            self._reject(
                f"'{node.name}' needs {expected} probabilities (2^{size}), "
                f"got {len(probabilities)}",
                node.initializer,
            )
            return

        for index, probability in enumerate(probabilities):
            if not 0 <= probability <= 1:
                self._reject(
                    f"the probability {probability} for value {index} of "
                    f"'{node.name}' is out of range (must be between 0 and 1)",
                    node.initializer,
                )
                return

        total = sum(probabilities)
        if total <= 0:
            self._reject(f"the probabilities for '{node.name}' cannot sum to zero",
                          node.initializer)
        elif abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            self._warn(
                f"the probabilities for '{node.name}' sum to {total}, not 1; "
                "the compiler will normalize them",
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


class _ArithmeticChecker(_Checker):
    """`+=`/`-=` and a quantum-times-quantum initializer are not general
    assignment -- both describe a fixed circuit shape over physical qubits,
    checked here the same way a qif condition is checked, not as ordinary
    expression typing."""

    def visit_AugAssign(self, node: AugAssign) -> None:
        self.generic_visit(node)

        target = node.target
        if not isinstance(target, Name):
            self._reject(
                "arithmetic assignment targets a whole quantum variable, "
                "not a single qubit",
                target,
            )
            return

        if self._require_quantum(target) is not None:
            self._check_value_shape(node.value, target)

    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        self.generic_visit(node)

        initializer = node.initializer
        if isinstance(initializer, BinaryOp) and initializer.op == "*":
            # A fresh declaration can never alias an operand by name, so
            # there is no self-reference to check here, unlike AugAssign.
            self._check_multiply_operands(initializer)

    def _check_value_shape(self, value: Expression, target: Name) -> None:
        if isinstance(value, Name):
            if isinstance(value.resolved_symbol, ParamDecl):
                self._reject_param_addend(value)
            elif self._require_quantum(value) is not None:
                self._reject_self_reference(target, (value,))
            return

        if isinstance(value, Index) and isinstance(
            value.base.resolved_symbol, ParamArrayDecl
        ):
            self._reject_param_addend(value)
            return

        if isinstance(value, BinaryOp) and value.op == "*":
            operands = self._check_multiply_operands(value)
            if operands is not None:
                self._reject_self_reference(target, operands)
            return

        if self._int(value) is None:
            self._reject(
                "this arithmetic assignment shape is not implemented yet; "
                "only adding/subtracting a quantum variable, a compile-time "
                "constant, or the product of two quantum variables is "
                "supported",
                value,
            )

    def _reject_param_addend(self, value: Expression) -> None:
        self._reject(
            "adding a runtime parameter to a quantum variable is not "
            "implemented yet; only a compile-time constant, another quantum "
            "variable, or the product of two quantum variables is "
            "supported here",
            value,
        )

    def _reject_self_reference(self, target: Name, operands: tuple[Name, ...]) -> None:
        # The adder/multiplier needs the addend to hold its original value
        # for the whole operation (an uncompute step, where there is one,
        # re-reads it) -- if the target is also an operand, its value has
        # already changed underneath that assumption.
        for operand in operands:
            if operand.name == target.name:
                self._reject(
                    f"'{target.name}' cannot appear on both sides of an "
                    "arithmetic assignment -- it is being changed, so it "
                    "cannot also be read as an unchanged operand",
                    operand,
                )
                return

    def _check_multiply_operands(self, value: BinaryOp) -> tuple[Name, Name] | None:
        operands: list[Name] = []
        for operand in (value.left, value.right):
            if not isinstance(operand, Name):
                self._reject(
                    "the product on the right of an arithmetic assignment "
                    "must multiply two whole quantum variables",
                    operand,
                )
                return None
            if self._require_quantum(operand) is None:
                return None
            operands.append(operand)
        return operands[0], operands[1]

    def _require_quantum(self, name: Name) -> Type | None:
        declared = _declared_type(name)
        if declared is None:
            self._reject(
                f"'{name.name}' is not a quantum variable; arithmetic "
                "assignment applies only to quantum variables",
                name,
            )
        return declared


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


# Comparisons other than "==" need a different circuit shape than a single
# ctrl_state pattern (an inequality has no single matching bit-pattern) and
# are deferred; listed explicitly so they get a clear message, not a generic one.
_UNSUPPORTED_QIF_COMPARISONS = frozenset({"<", ">", "<=", ">="})


class _QIfConditionChecker(_Checker):
    """A qif condition is not a general boolean expression that happens to
    need a quantum ingredient -- it is exclusively a description of a
    ctrl_state pattern over physical qubits. A classical value has no such
    pattern, so every clause, individually, must resolve to a quantum
    variable; that single rule covers both a purely classical condition and
    one that mixes classical and quantum clauses -- there is no second rule."""

    def visit_QIf(self, node: QIf) -> None:
        self.generic_visit(node)

        condition = node.condition
        if isinstance(condition, UnaryOp) and condition.op == "!":
            self._check_and_chain(condition.operand)
        else:
            self._check_and_chain(condition)

        self._check_no_overlap(node)

    def _check_and_chain(self, condition: Expression) -> None:
        if isinstance(condition, BinaryOp) and condition.op == "&&":
            self._check_and_chain(condition.left)
            self._check_clause(condition.right)
            return
        if isinstance(condition, BinaryOp) and condition.op == "||":
            self._reject("'||' in a qif condition is not implemented yet", condition)
            return
        self._check_clause(condition)

    def _check_clause(self, clause: Expression) -> None:
        # `!=` is sugar for a negated equality test -- same rules as `!(a==c)`.
        if isinstance(clause, BinaryOp) and clause.op == "!=":
            self._check_equality_clause(clause)
            return

        if isinstance(clause, UnaryOp) and clause.op == "!":
            inner = clause.operand
            if isinstance(inner, BinaryOp) and inner.op in ("&&", "||"):
                self._reject(
                    "a negated compound condition may only appear at the very "
                    "top of a qif condition; deeper nesting is not implemented yet",
                    clause,
                )
                return
            self._check_clause_shape(inner)
            return
        self._check_clause_shape(clause)

    def _check_clause_shape(self, clause: Expression) -> None:
        if isinstance(clause, Name):
            self._require_single_qubit(clause)
            return
        if isinstance(clause, Index):
            self._require_quantum(clause.base)
            return
        if isinstance(clause, BinaryOp) and clause.op in ("==", "!="):
            self._check_equality_clause(clause)
            return
        if isinstance(clause, BinaryOp) and clause.op in _UNSUPPORTED_QIF_COMPARISONS:
            self._reject(
                f"'{clause.op}' in a qif condition is not implemented yet", clause
            )
            return
        self._reject("this qif condition shape is not implemented yet", clause)

    def _check_equality_clause(self, clause: BinaryOp) -> None:
        name, literal = _split_equality(clause)
        if name is None:
            self._reject(
                "a qif equality test must compare a quantum variable to a "
                "constant",
                clause,
            )
            return

        declared = self._require_quantum(name)
        if declared is None:
            return

        size = qubit_count(declared)
        if size is None:
            # An unsized qint parameter; the lowering reports this on its own.
            return

        value = self._value(literal)
        if not isinstance(value, int):  # also accepts bool -- true/false fit in 1 qubit
            self._reject(
                "a qif equality test must compare against a compile-time constant",
                literal,
            )
            return

        largest = 2**size - 1
        if not 0 <= int(value) <= largest:
            self._reject(
                f"{int(value)} does not fit in '{name.name}', which holds {size} "
                f"qubit(s) (allowed: 0..{largest})",
                literal,
            )

    def _require_quantum(self, name: Name) -> Type | None:
        declared = _declared_type(name)
        if declared is None:
            self._reject(
                f"a qif condition may only reference quantum variables; "
                f"'{name.name}' is not one -- use if for a classical condition",
                name,
            )
        return declared

    def _require_single_qubit(self, name: Name) -> Type | None:
        declared = self._require_quantum(name)
        if declared is None:
            return None
        size = qubit_count(declared)
        if size is not None and size != 1:
            self._reject(
                f"'{name.name}' has {size} qubits; a bare qif condition needs "
                "exactly one -- index a specific bit, or compare the whole "
                "register with ==",
                name,
            )
            return None
        return declared

    def _check_no_overlap(self, node: QIf) -> None:
        condition_bits = self._collect_bits(node.condition)

        body_bits: set[tuple[str, int]] = set()
        for statement in node.body.statements:
            if isinstance(statement, ExprStatement) and isinstance(statement.expr, Call):
                for argument in statement.expr.args:
                    body_bits |= self._collect_bits(argument)

        overlap = condition_bits & body_bits
        if overlap:
            names = ", ".join(sorted({name for name, _ in overlap}))
            self._reject(
                f"a qif body may not modify a qubit its own condition tests "
                f"(here: {names})",
                node.body,
            )

    def _collect_bits(self, expression: Expression) -> set[tuple[str, int]]:
        if isinstance(expression, Name | Index):
            return self._touched_bits(expression) or set()
        bits: set[tuple[str, int]] = set()
        for child in iter_child_nodes(expression):
            bits |= self._collect_bits(child)
        return bits

    def _touched_bits(self, expression: Expression) -> set[tuple[str, int]] | None:
        if isinstance(expression, Index):
            declared = _declared_type(expression.base)
            if declared is None:
                return None
            try:
                index = const_int(expression.index)
            except ConstEvalError:
                return None
            if index is None:
                return None
            return {(expression.base.name, index)}

        if isinstance(expression, Name):
            declared = _declared_type(expression)
            if declared is None:
                return None
            size = qubit_count(declared)
            if size is None:
                return None
            return {(expression.name, bit) for bit in range(size)}

        return None


def _split_equality(clause: BinaryOp) -> tuple[Name | None, Expression]:
    """The quantum-variable side and the constant side of `==`/`!=`, in
    whichever order the programmer wrote them (`a == 2` or `2 == a`)."""
    if isinstance(clause.left, Name):
        return clause.left, clause.right
    if isinstance(clause.right, Name):
        return clause.right, clause.left
    return None, clause.right


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
    _ProbListChecker(bag).visit(ast)
    _QIfConditionChecker(bag).visit(ast)
    _ArithmeticChecker(bag).visit(ast)


def _check_affine(ast: Program, bag: DiagnosticBag) -> None:
    pass


def _error(bag: DiagnosticBag, message: str, span: Span) -> None:
    bag.error(message, line=span.start_line, column=span.start_col)


__all__ = ["analyze"]
