from __future__ import annotations

from slanq.ast_nodes import (
    Call,
    ClassicalDecl,
    Expression,
    ExprStatement,
    Index,
    Name,
    Program,
    QuantumDecl,
    Statement,
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
    IRModule,
    MeasurementOp,
    QubitBit,
    QubitOperand,
    QubitRef,
)
from slanq.visitor import NodeVisitor

MEASURE = "measure"

# Builtins the language defines but the compiler cannot build a circuit for yet.
UNIMPLEMENTED_BUILTINS: dict[str, str] = {
    # Outside a qif a phase shift is global and changes no measurement, so it
    # only becomes meaningful once qif exists.
    "phase": "the phase gate",
}

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


def lower_to_ir(ast: Program, bag: DiagnosticBag) -> IRModule:
    lowerer = _Lowerer(bag)
    lowerer.visit(ast)
    return lowerer.module


class _Lowerer(NodeVisitor):
    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        self.module = IRModule()
        self.qubits: dict[str, QubitRef] = {}

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

        value = (
            self._const_value(node.initializer)
            if isinstance(node.initializer, Expression)
            else None
        )
        # `isinstance` rather than `type(...) is int`: a qbool is initialized
        # with a bool, which is what the InitOp carries.
        if not isinstance(value, int):
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
        expr = node.expr
        if not isinstance(expr, Call):
            self._error("only call expressions are allowed as statements", node.span)
            return

        name = expr.callee.name
        if name in UNIMPLEMENTED_BUILTINS:
            self._error(
                f"{UNIMPLEMENTED_BUILTINS[name]} is not implemented yet; this is a "
                "limitation of the compiler, not an error in the program",
                node.span,
            )
            return

        signature = BUILTIN_SIGNATURES.get(name)
        if signature is None:
            self._error(
                "calling a process is not implemented yet; this is a limitation "
                "of the compiler, not an error in the program",
                node.span,
            )
            return

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
                    return
                params.append(argument)
                continue
            operand = self._operand(argument)
            if operand is None:
                return
            targets.append(operand)

        for row in _broadcast(signature, targets):
            self.module.body.ops.append(
                GateOp(span=node.span, name=name, targets=row, params=params)
            )

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


__all__ = ["lower_to_ir"]
