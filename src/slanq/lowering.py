from __future__ import annotations

from slanq.ast_nodes import (
    BinaryOp,
    Call,
    ClassicalDecl,
    Declaration,
    Expression,
    ExprStatement,
    Index,
    Name,
    ProbList,
    Program,
    QIf,
    QuantumDecl,
    Statement,
    UnaryOp,
    qubit_count,
)
from slanq.builtin import (
    BUILTIN_SIGNATURES,
    ArgKind,
    ConstEvalError,
    Signature,
    const_int,
    const_value,
)
from slanq.diagnostics import DiagnosticBag
from slanq.ir import (
    ClbitRef,
    GateOp,
    InitOp,
    IRBlock,
    IRModule,
    MeasurementOp,
    Op,
    PhaseOp,
    QIfClauseAncilla,
    QIfOp,
    QubitBit,
    QubitOperand,
    QubitRef,
)
from slanq.visitor import NodeVisitor

MEASURE = "measure"
PHASE = "phase"

# Statement kinds the lowering does not handle yet. Without this the generic
# traversal would walk straight past them and the construct would vanish from
# the circuit without a word.
UNIMPLEMENTED: dict[str, str] = {
    "Assign": "assignment",
    "AugAssign": "compound assignment",
    "If": "the if statement",
    "QIf": "the qif statement",
    "While": "the while loop",
    "For": "the for loop",
    "ProcessDef": "a process definition",
    "ParamDecl": "a param declaration",
    "ParamArrayDecl": "a param array declaration",
}

# Additional phrases for statements that a qif body specifically forbids, even
# though they lower fine at the top level -- distinct from UNIMPLEMENTED,
# which is for things not lowered anywhere yet.
QIF_BODY_ONLY_UNIMPLEMENTED: dict[str, str] = {
    "ClassicalDecl": "a classical declaration",
    "QuantumDecl": "a quantum declaration",
}


def lower_to_ir(ast: Program, bag: DiagnosticBag) -> IRModule:
    lowerer = _Lowerer(bag)
    lowerer.visit(ast)
    return lowerer.module


class _Lowerer(NodeVisitor):
    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        self.module = IRModule()
        self.qubits: dict[str, QubitRef] = {}
        self._declared_names: set[str] = set()
        self._ancilla_counter = 0

    def visit_Program(self, node: Program) -> None:
        # Needed only to pick ancilla names that cannot collide with anything
        # the program itself declares.
        self._declared_names = {
            statement.name
            for statement in node.statements
            if isinstance(statement, Declaration)
        }
        self.generic_visit(node)

    def generic_visit(self, node):
        if isinstance(node, Statement):
            self._unimplemented(node)
            return
        super().generic_visit(node)

    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        size = qubit_count(node.declared_type)
        if size is None:
            self._error(
                f"the type of '{node.name}' is not implemented yet; this is a "
                "limitation of the compiler, not an error in the program",
                node.span,
            )
            return

        value = self._init_value(node.initializer)
        if value is None:
            self._error(
                f"this way of initializing '{node.name}' is not implemented yet; "
                "this is a limitation of the compiler, not an error in the program",
                node.span,
            )
            return

        ref = QubitRef(name=node.name, size=size)
        self.qubits[node.name] = ref
        self.module.qubits.append(ref)
        self.module.body.ops.append(InitOp(span=node.span, target=ref, value=value))

    def visit_ClassicalDecl(self, node: ClassicalDecl) -> None:
        initializer = node.initializer
        if not (isinstance(initializer, Call) and initializer.callee.name == MEASURE):
            # A compile-time classical constant; it has no circuit representation.
            return

        source = self._measure_source(initializer)
        if source is None:
            return

        clbit = ClbitRef(name=node.name, size=source.size)
        self.module.clbits.append(clbit)
        self.module.body.ops.append(
            MeasurementOp(span=node.span, source=source, target=clbit)
        )

    def visit_ExprStatement(self, node: ExprStatement) -> None:
        ops = self._lower_gate_statement(node, in_qif=False)
        if ops is not None:
            self.module.body.ops.extend(ops)

    def visit_QIf(self, node: QIf) -> None:
        condition = self._lower_qif_condition(node.condition)
        if condition is None:
            return
        direct_qubits, ctrl_state, clause_ancillas, negated = condition

        top_ancilla = None
        if negated:
            top_ancilla = self._fresh_ancilla(node.span)
            if top_ancilla is None:
                return

        body = IRBlock()
        for statement in node.body.statements:
            if isinstance(statement, ExprStatement):
                ops = self._lower_gate_statement(statement, in_qif=True)
                if ops is not None:
                    body.ops.extend(ops)
            else:
                self._qif_body_unimplemented(statement)

        self.module.body.ops.append(
            QIfOp(
                span=node.span,
                direct_qubits=direct_qubits,
                ctrl_state=ctrl_state,
                clause_ancillas=clause_ancillas,
                negated=negated,
                ancilla=top_ancilla,
                body=body,
            )
        )

    def _lower_gate_statement(
        self, node: ExprStatement, *, in_qif: bool
    ) -> list[Op] | None:
        expr = node.expr
        if not isinstance(expr, Call):
            self._error("only call expressions are allowed as statements", node.span)
            return None

        name = expr.callee.name

        if name == MEASURE:
            # The else branch is unreachable: analysis already rejects a bare,
            # unassigned measure() before lowering runs.
            message = (
                "a qif body may only contain unitary operations; measurement is not one"
                if in_qif
                else "the result of measure() must be assigned"
            )
            self._error(message, node.span)
            return None

        if name == PHASE:
            return self._lower_phase(expr, node.span)

        signature = BUILTIN_SIGNATURES.get(name)
        if signature is None:
            self._error(
                "calling a process is not implemented yet; this is a limitation "
                "of the compiler, not an error in the program",
                node.span,
            )
            return None

        # Which argument is an angle and which is a qubit comes from the
        # signature, not from whether the argument happens to be constant: a
        # `param float` angle is not constant but is still an angle.
        targets: list[QubitOperand] = []
        params: list[Expression] = []
        for kind, argument in zip(signature.args, expr.args, strict=True):
            if kind is ArgKind.ANGLE:
                # A classical variable has no circuit representation yet, so an
                # angle that is not known at compile time cannot be emitted.
                if self._const_value(argument) is None:
                    self._error(
                        "an angle that is not known at compile time is not "
                        "implemented yet; this is a limitation of the compiler, "
                        "not an error in the program",
                        argument.span,
                    )
                    return None
                params.append(argument)
                continue
            operand = self._operand(argument)
            if operand is None:
                return None
            targets.append(operand)

        return [
            GateOp(span=node.span, name=name, targets=row, params=params)
            for row in _broadcast(signature, targets)
        ]

    def _lower_phase(self, expr: Call, span) -> list[Op] | None:
        (angle,) = expr.args
        if self._const_value(angle) is None:
            self._error(
                "an angle that is not known at compile time is not "
                "implemented yet; this is a limitation of the compiler, not "
                "an error in the program",
                angle.span,
            )
            return None
        return [PhaseOp(span=span, angle=angle)]

    def _qif_body_unimplemented(self, statement: Statement) -> None:
        kind = type(statement).__name__
        phrase = UNIMPLEMENTED.get(kind) or QIF_BODY_ONLY_UNIMPLEMENTED.get(kind, kind)
        self._error(
            f"{phrase} is not implemented inside a qif body yet; this is a "
            "limitation of the compiler, not an error in the program",
            statement.span,
        )

    def _lower_qif_condition(
        self, condition: Expression
    ) -> tuple[list[QubitOperand], int, list[QIfClauseAncilla], bool] | None:
        negated = isinstance(condition, UnaryOp) and condition.op == "!"
        chain = condition.operand if negated else condition

        qubits: list[QubitOperand] = []
        bits: list[int] = []
        clause_ancillas: list[QIfClauseAncilla] = []
        for clause in _flatten_and_chain(chain):
            result = self._lower_qif_clause(clause)
            if result is None:
                return None
            clause_qubits, clause_bits, clause_ancilla = result
            qubits.extend(clause_qubits)
            bits.extend(clause_bits)
            if clause_ancilla is not None:
                clause_ancillas.append(clause_ancilla)

        if negated and len(bits) == 1:
            bits[0] = 1 - bits[0]
            negated = False

        ctrl_state = 0
        for position, bit in enumerate(bits):
            ctrl_state |= bit << position

        return qubits, ctrl_state, clause_ancillas, negated

    def _lower_qif_clause(
        self, clause: Expression
    ) -> tuple[list[QubitOperand], list[int], QIfClauseAncilla | None] | None:
        if isinstance(clause, BinaryOp) and clause.op in ("==", "!="):
            return self._lower_equality_clause(clause, negated=clause.op == "!=")

        if isinstance(clause, UnaryOp) and clause.op == "!":
            inner = clause.operand
            if isinstance(inner, BinaryOp) and inner.op == "==":
                return self._lower_equality_clause(inner, negated=True)
            return self._lower_bare_clause(inner, negated=True)

        return self._lower_bare_clause(clause, negated=False)

    def _lower_bare_clause(
        self, target: Expression, *, negated: bool
    ) -> tuple[list[QubitOperand], list[int], QIfClauseAncilla | None] | None:
        if isinstance(target, Name):
            ref = self._register(target)
            if ref is None:
                return None
            return [QubitBit(ref=ref, index=0)], [0 if negated else 1], None

        assert isinstance(target, Index)
        ref = self._register(target.base)
        if ref is None:
            return None
        index = self._const_int(target.index)
        if index is None:
            self._error(
                "a quantum register index must be a compile-time integer",
                target.index.span,
            )
            return None
        return [QubitBit(ref=ref, index=index)], [0 if negated else 1], None

    def _lower_equality_clause(
        self, clause: BinaryOp, *, negated: bool
    ) -> tuple[list[QubitOperand], list[int], QIfClauseAncilla | None] | None:
        left, right = clause.left, clause.right
        name = left if isinstance(left, Name) else right
        literal = right if isinstance(left, Name) else left
        assert isinstance(name, Name)

        ref = self._register(name)
        if ref is None:
            return None
        value = self._const_value(literal)
        if not isinstance(value, int):
            return None
        constant = int(value)
        bits = [(constant >> position) & 1 for position in range(ref.size)]

        if not negated:
            return [ref], bits, None

        if ref.size == 1:
            return [QubitBit(ref=ref, index=0)], [1 - bits[0]], None

        # Wider: compute the positive match into its own ancilla and flip it.
        # The flipped ancilla is then just another 1-bit AND term -- negating
        # one clause needs no OR, only negating a whole subexpression does.
        ancilla = self._fresh_ancilla(clause.span)
        if ancilla is None:
            return None
        return (
            [QubitBit(ref=ancilla, index=0)],
            [1],
            QIfClauseAncilla(qubits=[ref], ctrl_state=constant, ancilla=ancilla),
        )

    def _fresh_ancilla(self, span) -> QubitRef | None:
        name = f"_ancilla_{self._ancilla_counter}"
        self._ancilla_counter += 1
        if name in self._declared_names:
            self._error(
                f"internal name '{name}' collides with a declaration in this "
                "program; rename it to compile this qif",
                span,
            )
            return None
        self._declared_names.add(name)
        return QubitRef(name=name, size=1, origin="ancilla")

    def _measure_source(self, call: Call) -> QubitRef | None:
        if len(call.args) != 1:
            self._error("measure() takes exactly one argument", call.span)
            return None

        argument = call.args[0]
        if not isinstance(argument, Name):
            self._error("measure() expects a quantum variable", call.span)
            return None

        ref = self.qubits.get(argument.name)
        if ref is None:
            self._error(f"'{argument.name}' is not a quantum variable", argument.span)
        return ref

    def _operand(self, expression: Expression) -> QubitOperand | None:
        if isinstance(expression, Index):
            return self._indexed_operand(expression)

        if not isinstance(expression, Name):
            self._error("expected a quantum variable", expression.span)
            return None

        ref = self._register(expression)
        if ref is None:
            return None

        # A single-qubit register is addressed bit-wise, so `circuit.h(q[0])` is
        # emitted rather than a broadcast over the whole register.
        return QubitBit(ref=ref, index=0) if ref.size == 1 else ref

    def _indexed_operand(self, expression: Index) -> QubitOperand | None:
        ref = self._register(expression.base)
        if ref is None:
            return None

        index = self._const_int(expression.index)
        if index is None:
            self._error(
                "a quantum register index must be a compile-time integer",
                expression.index.span,
            )
            return None

        return QubitBit(ref=ref, index=index)

    def _register(self, name: Name) -> QubitRef | None:
        ref = self.qubits.get(name.name)
        if ref is None:
            self._error(f"'{name.name}' is not a quantum variable", name.span)
        return ref

    def _init_value(self, initializer: Expression | ProbList) -> int | bool | list[float] | None:
        if isinstance(initializer, ProbList):
            return initializer.probabilities

        value = self._const_value(initializer)
        # bool is a subclass of int, so this also lets a qbool's True/False
        # through unchanged (`type(value) is int` would reject it).
        return value if isinstance(value, int) else None

    def _const_value(self, expression: Expression) -> int | float | bool | None:
        try:
            return const_value(expression)
        except ConstEvalError as exc:
            self._error(str(exc), expression.span)
            return None

    def _const_int(self, expression: Expression) -> int | None:
        try:
            return const_int(expression)
        except ConstEvalError as exc:
            self._error(str(exc), expression.span)
            return None

    def _unimplemented(self, node: Statement) -> None:
        kind = type(node).__name__
        self._error(
            f"{UNIMPLEMENTED.get(kind, kind)} is not implemented yet; this is a "
            "limitation of the compiler, not an error in the program",
            node.span,
        )

    def _error(self, message: str, span) -> None:
        self.bag.error(message, line=span.start_line, column=span.start_col)


def _broadcast(
    signature: Signature, targets: list[QubitOperand]
) -> list[list[QubitOperand]]:
    """One row per emitted gate. Qiskit spreads a register operand itself, so
    normally a single row is enough; CCX is the exception -- Qiskit refuses it,
    so the register is unrolled here into one gate per bit."""
    if signature.qiskit_broadcasts:
        return [targets]

    width = max(
        (target.size for target in targets if isinstance(target, QubitRef)), default=1
    )
    return [
        [
            QubitBit(ref=target, index=index) if isinstance(target, QubitRef) else target
            for target in targets
        ]
        for index in range(width)
    ]


def _flatten_and_chain(condition: Expression) -> list[Expression]:
    """Every clause of a `&&`-chain, left to right. Assumes analysis already
    rejected anything else that could appear here (`||`, deeper nesting)."""
    if isinstance(condition, BinaryOp) and condition.op == "&&":
        return [*_flatten_and_chain(condition.left), condition.right]
    return [condition]


__all__ = ["lower_to_ir"]
