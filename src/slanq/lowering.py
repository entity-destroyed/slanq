from __future__ import annotations

from slanq.ast_nodes import (
    Call,
    ClassicalDecl,
    Expression,
    ExprStatement,
    Literal,
    Name,
    Program,
    QBoolType,
    QuantumDecl,
    Type,
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


def lower_to_ir(ast: Program, bag: DiagnosticBag) -> IRModule:
    lowerer = _Lowerer(bag)
    lowerer.visit(ast)
    return lowerer.module


def _type_size(declared: Type) -> int | None:
    if isinstance(declared, QBoolType):
        return 1
    return None


class _Lowerer(NodeVisitor):
    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        self.module = IRModule()
        self.qubits: dict[str, QubitRef] = {}

    def visit_QuantumDecl(self, node: QuantumDecl) -> None:
        size = _type_size(node.declared_type)
        if size is None:
            self._error(f"unsupported quantum type for '{node.name}'", node.span)
            return

        if not isinstance(node.initializer, Literal):
            self._error(f"unsupported initializer for '{node.name}'", node.span)
            return

        ref = QubitRef(name=node.name, size=size)
        self.qubits[node.name] = ref
        self.module.qubits.append(ref)
        self.module.body.ops.append(
            InitOp(span=node.span, target=ref, value=node.initializer.value)
        )

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

        if expr.callee.name == MEASURE:
            self._error("the result of measure() must be assigned", node.span)
            return

        targets: list[QubitOperand] = []
        params: list[float] = []
        for argument in expr.args:
            if isinstance(argument, Literal):
                params.append(float(argument.value))
                continue
            operand = self._operand(argument)
            if operand is None:
                return
            targets.append(operand)

        self.module.body.ops.append(
            GateOp(span=node.span, name=expr.callee.name, targets=targets, params=params)
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
        if not isinstance(expression, Name):
            self._error("expected a quantum variable", expression.span)
            return None

        ref = self.qubits.get(expression.name)
        if ref is None:
            self._error(f"'{expression.name}' is not a quantum variable", expression.span)
            return None

        # A single-qubit register is addressed bit-wise, so `circuit.h(q[0])` is
        # emitted rather than a broadcast over the whole register.
        return QubitBit(ref=ref, index=0) if ref.size == 1 else ref

    def _error(self, message: str, span) -> None:
        self.bag.error(message, line=span.start_line, column=span.start_col)


__all__ = ["lower_to_ir"]
