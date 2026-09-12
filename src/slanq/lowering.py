from __future__ import annotations

from typing import Literal as TypingLiteral

from slanq.ast_nodes import (
    AmplitudeList,
    AugAssign,
    BinaryOp,
    Block,
    BoolType,
    Call,
    ClassicalDecl,
    Expression,
    ExprStatement,
    FloatType,
    For,
    If,
    Index,
    IntType,
    Literal,
    Name,
    ParamArrayDecl,
    ParamDecl,
    ProbList,
    ProcessDef,
    Program,
    QIf,
    QuantumDecl,
    RealtimeDecl,
    RealtimeIf,
    Statement,
    Type,
    UnaryOp,
    qubit_count,
)
from slanq.builtin import (
    BUILTIN_CONSTANTS,
    BUILTIN_GATES,
    BUILTIN_SIGNATURES,
    ArgKind,
    ConstEvalError,
    Signature,
    const_int,
    const_value,
)
from slanq.diagnostics import DiagnosticBag
from slanq.ir import (
    ArithmeticOp,
    ClassicalIfOp,
    ClbitRef,
    DeclareAncillaOp,
    DeclareRealtimeOp,
    GateOp,
    InitOp,
    IRBlock,
    IRModule,
    MeasurementOp,
    MultiplyOp,
    Op,
    ParamInfo,
    PhaseOp,
    QIfClauseAncilla,
    QIfOp,
    QubitBit,
    QubitOperand,
    QubitRef,
    QubitSlice,
    RealtimeIfOp,
    ResetOp,
)
from slanq.visitor import NodeVisitor

_PARAM_TYPE_NAMES: dict[type[Type], TypingLiteral["int", "float", "bool"]] = {
    IntType: "int",
    FloatType: "float",
    BoolType: "bool",
}

MEASURE = "measure"
PHASE = "phase"
RESET = "reset"

# Statement kinds the lowering does not handle yet. Without this the generic
# traversal would walk straight past them and the construct would vanish from
# the circuit without a word.
UNIMPLEMENTED: dict[str, str] = {
    "Assign": "assignment",
    "While": "the while loop",
}

# Additional phrases for statements that lower fine at the top level but a
# qif body specifically forbids -- distinct from UNIMPLEMENTED, which is for
# things not lowered anywhere yet. Everything named here is something a qif
# body could plausibly support with more implementation work later (a
# per-branch ancilla, a controlled adder, ...).
QIF_BODY_ONLY_UNIMPLEMENTED: dict[str, str] = {
    "ClassicalDecl": "a classical declaration",
    "QuantumDecl": "a quantum declaration",
    "AugAssign": "a compound assignment",
}

# Not a qif-specific restriction at all, and not "yet": a `param` is a
# whole-circuit, build-time object, so it can never mean anything nested
# inside a body, qif or otherwise -- there is no future qif-body work that
# would change this.
TOP_LEVEL_ONLY: dict[str, str] = {
    "ParamDecl": "a param declaration",
    "ParamArrayDecl": "a param array declaration",
}


# Above this many qubits a statevector simulator runs out of memory on an
# ordinary machine. Only a warning: real hardware has far more, and a program
# written for hardware is not wrong for being unsimulable.
MAX_SIMULABLE_QUBITS = 30


def lower_to_ir(ast: Program, bag: DiagnosticBag) -> IRModule:
    lowerer = _Lowerer(bag)
    lowerer.visit(ast)
    lowerer.warn_if_unsimulable()
    return lowerer.module


class _Lowerer(NodeVisitor):
    def __init__(self, bag: DiagnosticBag) -> None:
        self.bag = bag
        self.module = IRModule()
        self._block = self.module.body
        self.qubits: dict[str, QubitRef] = {}
        self._ancilla_counter = 0

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

        initializer = node.initializer
        if isinstance(initializer, BinaryOp) and initializer.op == "*":
            self._lower_multiply_decl(node, size, initializer)
            return

        if isinstance(initializer, AmplitudeList):
            self._lower_amplitude_decl(node, size, initializer)
            return

        value = self._init_value(initializer)
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
        self._block.ops.append(InitOp(span=node.span, target=ref, value=value))

    def visit_ClassicalDecl(self, node: ClassicalDecl) -> None:
        initializer = node.initializer
        if not (isinstance(initializer, Call) and initializer.callee.name == MEASURE):
            # A compile-time classical constant; it has no circuit representation.
            return

        self._unreachable("a measurement assigned to a build-time variable", node.span)
        source = self._measure_source(initializer)
        if source is None:
            return

        clbit = ClbitRef(name=node.name, size=source.size)
        self.module.clbits.append(clbit)
        self._block.ops.append(
            MeasurementOp(span=node.span, source=source, target=clbit)
        )

    def visit_ExprStatement(self, node: ExprStatement) -> None:
        ops = self._lower_gate_statement(node, in_qif=False)
        if ops is not None:
            self._block.ops.extend(ops)

    def visit_RealtimeDecl(self, node: RealtimeDecl) -> None:
        initializer = node.initializer
        if isinstance(initializer, Call) and initializer.callee.name == MEASURE:
            source = self._measure_source(initializer)
            if source is None:
                return
            clbit = ClbitRef(name=node.name, size=source.size)
            self.module.clbits.append(clbit)
            self._block.ops.append(
                MeasurementOp(span=node.span, source=source, target=clbit)
            )
            return

        if node.width is None:
            self._unreachable("a real-time variable of unknown width", node.span)
            return
        self._block.ops.append(
            DeclareRealtimeOp(
                span=node.span,
                name=node.name,
                size=node.width,
                value=initializer,
                is_bool=isinstance(node.declared_type, BoolType),
            )
        )

    def visit_RealtimeIf(self, node: RealtimeIf) -> None:
        body = self._lower_block(node.body)
        orelse = None if node.orelse is None else self._lower_block(node.orelse)
        self._block.ops.append(
            RealtimeIfOp(
                span=node.span, condition=node.condition, body=body, orelse=orelse
            )
        )

    def visit_If(self, node: If) -> None:
        body = self._lower_block(node.body)
        orelse = None if node.orelse is None else self._lower_block(node.orelse)
        self._block.ops.append(
            ClassicalIfOp(
                span=node.span,
                condition=node.condition,
                body=body,
                orelse=orelse,
            )
        )

    def _lower_block(self, block: Block) -> IRBlock:
        inner = IRBlock()
        outer, self._block = self._block, inner
        try:
            for statement in block.statements:
                self.visit(statement)
        finally:
            self._block = outer
        return inner

    def visit_ProcessDef(self, node: ProcessDef) -> None:
        # A template, not code: the expansion pass copied its body into every
        # call site, so the definition itself contributes no operations.
        return

    def visit_For(self, node: For) -> None:
        self._unreachable("a for loop", node.span)

    def visit_ParamDecl(self, node: ParamDecl) -> None:
        type_name = _PARAM_TYPE_NAMES.get(type(node.declared_type))
        if type_name is None:
            self._unreachable("a param of an unsupported type", node.span)
            return
        self.module.params.append(
            ParamInfo(name=node.name, kind="scalar", size=None, type_name=type_name)
        )

    def visit_ParamArrayDecl(self, node: ParamArrayDecl) -> None:
        type_name = _PARAM_TYPE_NAMES.get(type(node.declared_type))
        if type_name is None:
            self._unreachable("a param array of an unsupported type", node.span)
            return
        self.module.params.append(
            ParamInfo(
                name=node.name,
                kind="array",
                size=node.size,
                type_name=type_name,
            )
        )

    def visit_AugAssign(self, node: AugAssign) -> None:
        if not isinstance(node.target, Name):
            self._unreachable("an arithmetic target that is not a variable", node.span)
            return
        target = self._register(node.target)
        if target is None:
            return
        subtract = node.op == "-="

        value = node.value
        if isinstance(value, BinaryOp) and value.op == "*":
            self._lower_multiply_augassign(node.span, target, value, subtract)
            return

        helper = self._fresh_ancilla(node.span)

        if isinstance(value, Name):
            addend_ref = self._register(value)
            if addend_ref is None:
                return
            addend = self._pad_to_width(addend_ref, target.size, node.span)
            self._block.ops.append(
                ArithmeticOp(
                    span=node.span,
                    target=target,
                    addend=addend,
                    helper=helper,
                    subtract=subtract,
                    encode_constant=None,
                )
            )
            return

        constant = self._const_int(value)
        if constant is None:
            self._unreachable("an addend that is not a whole constant", node.span)
            return
        ancilla = self._fresh_ancilla(node.span, size=target.size)
        self._block.ops.append(
            ArithmeticOp(
                span=node.span,
                target=target,
                addend=[ancilla],
                helper=helper,
                subtract=subtract,
                encode_constant=constant % (1 << target.size),
            )
        )

    def visit_QIf(self, node: QIf) -> None:
        condition = self._lower_qif_condition(node.condition)
        if condition is None:
            return
        direct_qubits, ctrl_state, clause_ancillas, negated = condition

        top_ancilla = None
        if negated:
            top_ancilla = self._fresh_ancilla(node.span)

        body = self._lower_qif_body(node.body)

        self._block.ops.append(
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

    def _lower_qif_body(self, block: Block) -> IRBlock:
        result = IRBlock()
        for statement in block.statements:
            if isinstance(statement, ExprStatement):
                ops = self._lower_gate_statement(statement, in_qif=True)
                if ops is not None:
                    result.ops.extend(ops)
            elif isinstance(statement, If):
                branch = ClassicalIfOp(
                    span=statement.span,
                    condition=statement.condition,
                    body=self._lower_qif_body(statement.body),
                    orelse=None
                    if statement.orelse is None
                    else self._lower_qif_body(statement.orelse),
                )
                result.ops.append(branch)
            else:
                self._qif_body_unimplemented(statement)
        return result

    def _lower_gate_statement(
        self, node: ExprStatement, *, in_qif: bool
    ) -> list[Op] | None:
        expr = node.expr
        if not isinstance(expr, Call):
            self._error("only call expressions are allowed as statements", node.span)
            return None

        name = expr.callee.name

        signature = BUILTIN_SIGNATURES.get(name)
        if signature is None:
            if isinstance(expr.callee.resolved_symbol, ProcessDef):
                self._error(
                    f"internal error: the call to '{name}' was not expanded "
                    "before lowering. This is a bug in the compiler, not in "
                    "the source program.",
                    node.span,
                )
            else:
                self._error(f"'{name}' is not a gate", node.span)
            return None

        if len(expr.args) != len(signature.args):
            self._unreachable(
                f"a call to '{name}' with {len(expr.args)} argument(s) instead "
                f"of {len(signature.args)}",
                node.span,
            )
            return None

        if in_qif and _is_non_unitary_quantum(name, signature):
            if signature.returns is None:
                self._error(_not_unitary(f"'{name}'"), node.span)
            return None

        if name == MEASURE:
            self._unreachable("an unassigned measure()", node.span)
            return None

        if name == PHASE:
            return self._lower_phase(expr, node.span)

        if name == RESET:
            return self._lower_reset(expr, node.span)

        # Which argument is an angle and which is a qubit comes from the
        # signature, not from whether the argument happens to be constant: a
        # `param float` angle is not constant but is still an angle.
        targets: list[QubitOperand] = []
        params: list[Expression] = []
        for kind, argument in zip(signature.args, expr.args, strict=True):
            if kind is ArgKind.ANGLE:
                if not self._is_renderable_angle(argument):
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
        if not self._is_renderable_angle(angle):
            return None
        return [PhaseOp(span=span, angle=angle)]

    def _lower_reset(self, expr: Call, span) -> list[Op] | None:
        (target,) = expr.args
        operand = self._operand(target)
        if operand is None:
            return None
        return [ResetOp(span=span, target=operand)]

    def _is_renderable_angle(self, expression: Expression) -> bool:
        """Whether `expression` can reach the generated file as an angle --
        either it has a compile-time value, or it is a `param` expression
        that codegen can print as a `Parameter`-valued Python expression."""
        try:
            value = const_value(expression)
        except ConstEvalError as exc:
            self._error(str(exc), expression.span)
            return False
        if isinstance(value, complex):
            self._error(
                "a complex number cannot be used as an angle; only int or "
                "float are supported here",
                expression.span,
            )
            return False
        if type(value) is bool or (value is None and _is_boolean_param_expression(expression)):
            self._error(
                "a bool value cannot be used as an angle; only int or float "
                "are supported here",
                expression.span,
            )
            return False
        if value is None and not _is_param_expression(expression):
            self._error(
                "an angle that is not known at compile time is not "
                "implemented yet; this is a limitation of the compiler, not "
                "an error in the program",
                expression.span,
            )
            return False
        return True

    def _qif_body_unimplemented(self, statement: Statement) -> None:
        kind = type(statement).__name__
        if isinstance(statement, RealtimeIf):
            self._error(_not_unitary("a real-time branch"), statement.span)
            return
        if _measured_initializer(statement):
            self._error(_not_unitary("measurement"), statement.span)
            return
        if kind in TOP_LEVEL_ONLY:
            self._error(
                f"{TOP_LEVEL_ONLY[kind]} is only allowed at the top level of "
                "a program, not inside a qif body",
                statement.span,
            )
            return
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
            if clause_ancilla is not None:
                twin = next(
                    (
                        existing
                        for existing in clause_ancillas
                        if existing.qubits == clause_ancilla.qubits
                        and existing.ctrl_state == clause_ancilla.ctrl_state
                    ),
                    None,
                )
                if twin is None:
                    clause_ancillas.append(clause_ancilla)
                else:
                    # The same negated test twice: one ancilla already holds
                    # the answer, so this clause points at that one and the
                    # merge below drops it as a repetition.
                    self._discard_ancilla(clause_ancilla.ancilla)
                    clause_qubits = [QubitBit(ref=twin.ancilla, index=0)]
            qubits.extend(clause_qubits)
            bits.extend(clause_bits)

        merged = self._merge_condition(
            qubits, bits, clause_ancillas, condition.span, negated=negated
        )
        if merged is None:
            return None
        qubits, bits = merged

        if negated and len(bits) == 1:
            bits[0] = 1 - bits[0]
            negated = False

        ctrl_state = 0
        for position, bit in enumerate(bits):
            ctrl_state |= bit << position

        return qubits, ctrl_state, clause_ancillas, negated

    def _merge_condition(
        self,
        qubits: list[QubitOperand],
        bits: list[int],
        clause_ancillas: list[QIfClauseAncilla],
        span,
        *,
        negated: bool,
    ) -> tuple[list[QubitOperand], list[int]] | None:
        """One entry per physical qubit. Two clauses can land on the same qubit
        -- the same test written twice, or `a == 2 && a[0]` -- and then either
        they agree, in which case the second says nothing, or they disagree, in
        which case no state satisfies the condition."""
        required: dict[tuple[str, int], int] = {}
        kept: list[tuple[QubitOperand, list[int]]] = []
        repeated = False

        position = 0
        for operand in qubits:
            physical = _condition_qubits(operand)
            wanted = bits[position : position + len(physical)]
            position += len(physical)

            for qubit, bit in zip(physical, wanted, strict=True):
                if required.get(qubit, bit) != bit:
                    self._reject_impossible(
                        f"'{qubit[0]}' would have to hold two different values "
                        "at once",
                        span,
                        negated=negated,
                    )
                    return None

            seen = {qubit for qubit in physical if qubit in required}
            if seen:
                repeated = True
                if len(seen) == len(physical):
                    continue
                kept = [
                    (entry, entry_bits)
                    for entry, entry_bits in kept
                    if not seen & set(_condition_qubits(entry))
                ]

            required.update(zip(physical, wanted, strict=True))
            kept.append((operand, wanted))

        # A negated clause excludes exactly one combination of its register,
        # so it fixes no single qubit and cannot be part of the map above --
        # but if the rest of the condition forces precisely the combination it
        # excludes, nothing is left to satisfy it.
        for clause_ancilla in clause_ancillas:
            excluded = [
                (qubit, (clause_ancilla.ctrl_state >> position) & 1)
                for entry in clause_ancilla.qubits
                for position, qubit in enumerate(_condition_qubits(entry))
            ]
            if excluded and all(required.get(qubit) == bit for qubit, bit in excluded):
                name = excluded[0][0][0]
                self._reject_impossible(
                    f"'{name}' is required to hold exactly the value another "
                    "part of the condition excludes",
                    span,
                    negated=negated,
                )
                return None

        if repeated:
            self._warn(
                "this qif condition tests the same qubit more than once; the "
                "repeated test is dropped",
                span,
            )
        return (
            [operand for operand, _ in kept],
            [bit for _, entry_bits in kept for bit in entry_bits],
        )

    def _reject_impossible(self, detail: str, span, *, negated: bool) -> None:
        if negated:
            self._error(
                f"this qif condition is always true, because the test it "
                f"negates can never be: {detail}",
                span,
            )
            return
        self._error(f"this qif condition can never be true: {detail}", span)

    def _discard_ancilla(self, ref: QubitRef) -> None:
        self.module.body.ops = [
            op
            for op in self.module.body.ops
            if not (isinstance(op, DeclareAncillaOp) and op.ref is ref)
        ]

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

        if not isinstance(target, Index):
            self._unreachable("a qif condition that is not a qubit", target.span)
            return None
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
        if not isinstance(name, Name):
            self._unreachable(
                "a qif equality test with no quantum operand", clause.span
            )
            return None

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
        return (
            [QubitBit(ref=ancilla, index=0)],
            [1],
            QIfClauseAncilla(qubits=[ref], ctrl_state=constant, ancilla=ancilla),
        )

    def _fresh_ancilla(self, span, size: int = 1) -> QubitRef:
        """Allocates a register and declares it in the same breath, so the
        declaration exists exactly once and stands before every use."""
        name = f"_ancilla_{self._ancilla_counter}"
        self._ancilla_counter += 1
        ref = QubitRef(name=name, size=size)
        self.module.body.ops.append(DeclareAncillaOp(span=span, ref=ref))
        return ref

    def _pad_to_width(
        self, ref: QubitRef, width: int, span
    ) -> list[QubitOperand]:
        """`ref`'s bits, concatenated with a fresh 0-ancilla if it is
        narrower than `width`, or sliced to its low bits if it is wider --
        the CDKM/HRS adder and multiplier both preserve their `a`/`b`
        inputs exactly, so a 0-ancilla among them needs no uncompute."""
        if ref.size == width:
            return [ref]
        if ref.size > width:
            return [QubitSlice(ref=ref, size=width)]
        padding = self._fresh_ancilla(span, size=width - ref.size)
        return [ref, padding]

    def _lower_multiply_operands(
        self, span, left: Expression, right: Expression
    ) -> tuple[list[QubitOperand], list[QubitOperand]] | None:
        if not (isinstance(left, Name) and isinstance(right, Name)):
            self._unreachable("a product of something other than variables", span)
            return None
        left_ref = self._register(left)
        right_ref = self._register(right)
        if left_ref is None or right_ref is None:
            return None

        width = max(left_ref.size, right_ref.size)
        return (
            self._pad_to_width(left_ref, width, span),
            self._pad_to_width(right_ref, width, span),
        )

    def _lower_amplitude_decl(
        self, node: QuantumDecl, size: int, initializer: AmplitudeList
    ) -> None:
        amplitudes: list[complex | float] = []
        for element in initializer.elements:
            value = self._const_value(element)
            if value is None or isinstance(value, bool):
                self._unreachable("an amplitude that is not a number", element.span)
                return
            amplitudes.append(value)

        ref = QubitRef(name=node.name, size=size)
        self.qubits[node.name] = ref
        self.module.qubits.append(ref)
        self._block.ops.append(
            InitOp(
                span=node.span,
                target=ref,
                value=amplitudes,
                is_amplitude=True,
                value_expressions=initializer.elements,
            )
        )

    def _lower_multiply_decl(
        self, node: QuantumDecl, size: int, initializer: BinaryOp
    ) -> None:
        operands = self._lower_multiply_operands(
            node.span, initializer.left, initializer.right
        )
        if operands is None:
            return
        left, right = operands
        helper = self._fresh_ancilla(node.span)

        ref = QubitRef(name=node.name, size=size)
        self.qubits[node.name] = ref
        self.module.qubits.append(ref)
        self._block.ops.append(
            MultiplyOp(
                span=node.span,
                left=left,
                right=right,
                product=ref,
                helper=helper,
                inverse=False,
            )
        )

    def _lower_multiply_augassign(
        self, span, target: QubitRef, value: BinaryOp, subtract: bool
    ) -> None:
        operands = self._lower_multiply_operands(span, value.left, value.right)
        if operands is None:
            return
        left, right = operands
        width = sum(_operand_width(operand) for operand in left)

        # Sized 2*width so the product is never truncated before the '+='
        # below applies target's own, possibly narrower, width rule to it.
        temp = self._fresh_ancilla(span, size=2 * width)
        multiply_helper = self._fresh_ancilla(span)
        self._block.ops.append(
            MultiplyOp(
                span=span,
                left=left,
                right=right,
                product=temp,
                helper=multiply_helper,
                inverse=False,
            )
        )

        add_helper = self._fresh_ancilla(span)
        addend = self._pad_to_width(temp, target.size, span)
        self._block.ops.append(
            ArithmeticOp(
                span=span,
                target=target,
                addend=addend,
                helper=add_helper,
                subtract=subtract,
                encode_constant=None,
            )
        )

        # a and b (`left`/`right`) are untouched by the multiplier, so running
        # it again as its own inverse cleanly zeroes the temporary product.
        self._block.ops.append(
            MultiplyOp(
                span=span,
                left=left,
                right=right,
                product=temp,
                helper=multiply_helper,
                inverse=True,
            )
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

    def _init_value(self, initializer: Expression | ProbList) -> int | bool | list[float] | None:
        if isinstance(initializer, ProbList):
            return initializer.probabilities

        value = self._const_value(initializer)
        # bool is a subclass of int, so this also lets a qbool's True/False
        # through unchanged (`type(value) is int` would reject it).
        return value if isinstance(value, int) else None

    def _const_value(self, expression: Expression) -> int | float | bool | complex | None:
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

    def warn_if_unsimulable(self) -> None:
        """Counted here rather than in analysis because the ancillas the
        compiler allocates are qubits too, and they exist only once lowering
        has run. Counted over the whole circuit, not per register: what a
        simulator cannot hold is the total, however it is divided up."""
        total = sum(ref.size for ref in self.module.qubits) + sum(
            op.ref.size
            for op in self.module.body.ops
            if isinstance(op, DeclareAncillaOp)
        )
        if total > MAX_SIMULABLE_QUBITS:
            self.bag.warning(
                f"this program needs {total} qubits, more than the "
                f"{MAX_SIMULABLE_QUBITS} a simulator usually holds; it can "
                "still be built and run on hardware"
            )

    def _error(self, message: str, span) -> None:
        self.bag.error(message, line=span.start_line, column=span.start_col)

    def _warn(self, message: str, span) -> None:
        self.bag.warning(message, line=span.start_line, column=span.start_col)

    def _unreachable(self, what: str, span) -> None:
        """A shape analysis rejects, reached anyway because lowering runs even
        after an error. The diagnostic that rejected it already stands, so
        nothing is added -- unless the bag is empty, which means analysis let
        this through and the invariant is genuinely broken."""
        if not self.bag.has_errors:
            self._error(
                f"internal error: {what} reached lowering. This is a bug in "
                "the compiler, not in the source program.",
                span,
            )


def _not_unitary(what: str) -> str:
    return f"a qif body may only contain unitary operations; {what} is not one"


def _is_non_unitary_quantum(name: str, signature: Signature) -> bool:
    """Whether `name` acts on qubits without being unitary -- example: `measure` and
    `reset`. """
    return name not in BUILTIN_GATES and any(
        kind is not ArgKind.ANGLE for kind in signature.args
    )


def _measured_initializer(node: Statement) -> bool:
    """A classical declaration whose value comes from a non-unitary quantum
    builtin. Its statement kind says `ClassicalDecl`, but what a qif body
    rejects about it is the measurement, not the declaration."""
    if not isinstance(node, ClassicalDecl | RealtimeDecl) or not isinstance(
        node.initializer, Call
    ):
        return False
    signature = BUILTIN_SIGNATURES.get(node.initializer.callee.name)
    return signature is not None and _is_non_unitary_quantum(
        node.initializer.callee.name, signature
    )


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


def _condition_qubits(operand: QubitOperand) -> list[tuple[str, int]]:
    """`operand`'s physical qubits, in the order its condition bits are listed.
    A whole register contributes its qubits in Qiskit order; a `QubitBit` holds
    a Slanq index, which counts from the most significant qubit, so it mirrors
    -- each shape the same way the code generator reads it."""
    if isinstance(operand, QubitBit):
        return [(operand.ref.name, operand.ref.size - 1 - operand.index)]
    if isinstance(operand, QubitSlice):
        return [(operand.ref.name, index) for index in range(operand.size)]
    return [(operand.name, index) for index in range(operand.size)]


def _operand_width(operand: QubitOperand) -> int:
    return 1 if isinstance(operand, QubitBit) else operand.size


# Qiskit's ParameterExpression only overloads +, -, *, /, ** and unary -; `%`,
# the bitwise operators and float()/int() conversions all raise TypeError on
# it, so an expression involving a param cannot use them, unlike a fully
# constant one.
_PARAM_EXPRESSION_BINARY_OPS = frozenset({"+", "-", "*", "/", "**"})


def _is_param_expression(expression: Expression) -> bool:
    """Whether `expression` can reach the generated file as a `Parameter`-
    valued Python expression even though it has no compile-time value."""
    if isinstance(expression, Literal):
        return True
    if isinstance(expression, Name):
        return expression.name in BUILTIN_CONSTANTS or isinstance(
            expression.resolved_symbol, ParamDecl
        )
    if isinstance(expression, Index):
        if not isinstance(expression.base.resolved_symbol, ParamArrayDecl):
            return False
        try:
            return const_int(expression.index) is not None
        except ConstEvalError:
            return False
    if isinstance(expression, UnaryOp):
        return expression.op == "-" and _is_param_expression(expression.operand)
    if isinstance(expression, BinaryOp):
        return (
            expression.op in _PARAM_EXPRESSION_BINARY_OPS
            and _is_param_expression(expression.left)
            and _is_param_expression(expression.right)
        )
    return False


def _is_boolean_param_expression(expression: Expression) -> bool:
    """Whether a param expression involves a `param bool` -- arithmetic never
    produces `bool` in Python, so this only needs to check leaves, not the
    combinators (`+ - * / ** -`) that hold the tree together."""
    if isinstance(expression, Name):
        return isinstance(expression.resolved_symbol, ParamDecl) and isinstance(
            expression.resolved_symbol.declared_type, BoolType
        )
    if isinstance(expression, Index):
        return isinstance(
            expression.base.resolved_symbol, ParamArrayDecl
        ) and isinstance(expression.base.resolved_symbol.declared_type, BoolType)
    if isinstance(expression, UnaryOp):
        return _is_boolean_param_expression(expression.operand)
    if isinstance(expression, BinaryOp):
        return _is_boolean_param_expression(
            expression.left
        ) or _is_boolean_param_expression(expression.right)
    return False


__all__ = ["lower_to_ir"]
