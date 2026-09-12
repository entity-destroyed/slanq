from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import (
    AugAssign,
    Block,
    Call,
    ExprStatement,
    For,
    Index,
    IntType,
    Literal,
    LoopVarDecl,
    Name,
    ProcessDef,
    ProcParam,
    Program,
    QBoolType,
    QIntType,
    QuantumDecl,
    Span,
)
from slanq.builtin import const_value
from slanq.diagnostics import DiagnosticBag
from slanq.ir import (
    ArithmeticOp,
    ClassicalIfOp,
    ClbitRef,
    DeclareAncillaOp,
    GateOp,
    InitOp,
    IRModule,
    MeasurementOp,
    MultiplyOp,
    Op,
    PhaseOp,
    QIfClauseAncilla,
    QIfOp,
    QubitBit,
    QubitRef,
    QubitSlice,
    ResetOp,
)
from slanq.lowering import lower_to_ir

BuildAst = Callable[[str], Program]
LowerSource = Callable[[str], tuple[IRModule, DiagnosticBag]]


@pytest.fixture
def lower(build_ast: BuildAst) -> LowerSource:
    def _lower(source: str) -> tuple[IRModule, DiagnosticBag]:
        ast = build_ast(source)
        bag = DiagnosticBag()
        analyze(ast, bag)
        return lower_to_ir(ast, bag), bag

    return _lower


def test_quantum_decl_creates_register_and_init(lower: LowerSource) -> None:
    module, bag = lower("qbool q = false;")
    assert not bag.has_errors

    (register,) = module.qubits
    assert register == QubitRef(name="q", size=1)

    (init,) = module.body.ops
    assert isinstance(init, InitOp)
    assert init.target == register
    assert init.value is False


def test_gate_call_becomes_gate_op(lower: LowerSource) -> None:
    module, bag = lower("qbool q = false; H(q);")
    assert not bag.has_errors

    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert gate.name == "H"
    assert gate.params == []


def test_single_qubit_register_is_addressed_bitwise(lower: LowerSource) -> None:
    module, _ = lower("qbool q = false; H(q);")
    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert gate.targets == [QubitBit(ref=module.qubits[0], index=0)]


def test_measurement_creates_clbit_and_op(lower: LowerSource) -> None:
    module, bag = lower("qbool q = false; int result = measure(q);")
    assert not bag.has_errors

    (clbit,) = module.clbits
    assert clbit == ClbitRef(name="result", size=1)

    measurement = module.body.ops[1]
    assert isinstance(measurement, MeasurementOp)
    assert measurement.source == module.qubits[0]
    assert measurement.target == clbit


def test_reset_of_a_whole_register_becomes_one_op(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; reset(a);")
    assert not bag.has_errors

    reset = module.body.ops[1]
    assert isinstance(reset, ResetOp)
    assert reset.target == module.qubits[0]


def test_reset_of_one_qubit_keeps_the_index(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; reset(a[1]);")
    assert not bag.has_errors

    reset = module.body.ops[1]
    assert isinstance(reset, ResetOp)
    assert reset.target == QubitBit(ref=module.qubits[0], index=1)


_NOT_UNITARY = "a qif body may only contain unitary operations; {} is not one"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("reset(b);", _NOT_UNITARY.format("'reset'")),
        ("measure(b);", "the result of 'measure' must be assigned"),
        ("int r = measure(b);", _NOT_UNITARY.format("measurement")),
    ],
    ids=["reset", "bare measure", "assigned measure"],
)
def test_a_qif_body_rejects_a_non_unitary_operation(
    lower: LowerSource, body: str, message: str
) -> None:
    """A bare measure is rejected by analysis as an unassigned result, so the
    qif rule stays silent there and one mistake still yields one message."""
    _, bag = lower(f"qint<2> a = 0; qint<2> b = 0; qif(a == 2) {{ {body} }}")
    (diagnostic,) = bag.errors
    assert diagnostic.message == message


def test_numeric_arguments_become_params(lower: LowerSource) -> None:
    module, bag = lower("qbool q = false; RX(90, q);")
    assert not bag.has_errors

    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert [const_value(param) for param in gate.params] == [90]
    assert gate.targets == [QubitBit(ref=module.qubits[0], index=0)]


def test_classical_constant_produces_no_ir(lower: LowerSource) -> None:
    module, bag = lower("int x = 42;")
    assert not bag.has_errors
    assert module.clbits == []
    assert module.body.ops == []


def test_bare_measure_statement_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; measure(q);")
    assert bag.has_errors
    assert "must be assigned" in bag.errors[0].message


def test_gate_on_unknown_variable_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; H(missing);")
    assert bag.has_errors


def test_spans_are_carried_into_the_ir(lower: LowerSource) -> None:
    module, _ = lower("qbool q = false;\nH(q);")
    gate = module.body.ops[1]
    assert gate.span.start_line == 2


def test_mvp_a_lowers_completely(lower: LowerSource, mvp_a_source: str) -> None:
    module, bag = lower(mvp_a_source)
    assert not bag.has_errors
    assert [type(op).__name__ for op in module.body.ops] == [
        "InitOp",
        "GateOp",
        "MeasurementOp",
    ]
    assert len(module.qubits) == 1
    assert len(module.clbits) == 1


def test_qint_register_gets_its_declared_size(lower: LowerSource) -> None:
    module, bag = lower("qint<3> a = 0;")
    assert not bag.has_errors
    assert module.qubits[0] == QubitRef(name="a", size=3)


def test_index_lowers_with_the_slanq_index(lower: LowerSource) -> None:
    """The IR stays in the user's index space; codegen does the mirroring."""
    module, bag = lower("qint<3> a = 0; X(a[1]);")
    assert not bag.has_errors
    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert gate.targets == [QubitBit(ref=module.qubits[0], index=1)]


def test_two_qubit_gate_keeps_operand_order(lower: LowerSource) -> None:
    module, bag = lower("qint<2> q = 0; CX(q[0], q[1]);")
    assert not bag.has_errors
    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    register = module.qubits[0]
    assert gate.targets == [
        QubitBit(ref=register, index=0),
        QubitBit(ref=register, index=1),
    ]


def test_multi_qubit_measurement_sizes_the_register(lower: LowerSource) -> None:
    module, bag = lower("qint<3> a = 0; int r = measure(a);")
    assert not bag.has_errors
    assert module.clbits[0] == ClbitRef(name="r", size=3)


def test_whole_register_target_is_kept_whole(lower: LowerSource) -> None:
    """B6: a gate applied to a multi-qubit register broadcasts."""
    module, bag = lower("qint<3> a = 0; H(a);")
    assert not bag.has_errors
    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert gate.targets == [module.qubits[0]]


def test_mvp_b_lowers_completely(lower: LowerSource, mvp_b_source: str) -> None:
    module, bag = lower(mvp_b_source)
    assert not bag.has_errors
    assert [type(op).__name__ for op in module.body.ops] == [
        "InitOp",
        "GateOp",
        "GateOp",
        "MeasurementOp",
    ]


def test_unimplemented_statement_does_not_vanish(lower: LowerSource) -> None:
    """Before the guard this produced no IR and no diagnostic at all: the
    generated circuit simply never performed the assignment."""
    _, bag = lower("qbool q = false; q = true;")
    assert bag.has_errors
    assert "not implemented yet" in bag.errors[0].message


def test_unimplemented_statement_names_the_construct(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; while(true) { X(q); }")
    assert "while loop" in bag.errors[0].message


def test_unimplemented_message_blames_the_compiler(lower: LowerSource) -> None:
    """A quantum variable initialized from a `param`: a `param` is a runtime
    `Parameter`, never a compile-time value, so this stays a limitation, not
    a crash -- unlike a classical constant, see
    test_classical_name_initializes_a_quantum_variable below."""
    _, bag = lower("param int t; qint<2> b = t;")
    assert bag.has_errors
    assert "limitation of the compiler" in bag.errors[0].message


def test_classical_name_initializes_a_quantum_variable(lower: LowerSource) -> None:
    module, bag = lower("int t = 2; qint<2> b = t;")
    assert not bag.has_errors
    init = module.body.ops[-1]
    assert isinstance(init, InitOp)
    assert init.value == 2


def test_ccx_on_registers_becomes_one_gate_per_bit(lower: LowerSource) -> None:
    """Qiskit refuses a register operand for ccx, so the lowering unrolls it."""
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; qint<2> c = 0; CCX(a, b, c);")
    assert not bag.has_errors

    gates = [op for op in module.body.ops if isinstance(op, GateOp)]
    assert len(gates) == 2
    assert all(isinstance(target, QubitBit) for gate in gates for target in gate.targets)
    assert [target.index for target in gates[0].targets] == [0, 0, 0]
    assert [target.index for target in gates[1].targets] == [1, 1, 1]


def test_two_qubit_gate_stays_a_single_op(lower: LowerSource) -> None:
    """Qiskit spreads a register operand itself, so one op is enough."""
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; CX(a, b);")
    assert not bag.has_errors
    assert len([op for op in module.body.ops if isinstance(op, GateOp)]) == 1


def test_angle_position_comes_from_the_signature(lower: LowerSource) -> None:
    """Not from whether the argument is constant: `q` is constant-free and a
    qubit, `PI` is neither -- only the signature tells them apart."""
    module, bag = lower("qbool q = false; RX(PI, q);")
    assert not bag.has_errors

    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert len(gate.params) == 1
    assert gate.targets == [QubitBit(ref=module.qubits[0], index=0)]


def test_runtime_angle_is_reported_as_a_limitation(lower: LowerSource) -> None:
    """A classical constant (`float t = 1.0;`) is resolvable at compile time
    (see test_classical_name_is_a_renderable_angle below) -- a genuinely
    runtime value, like a measurement result, is what stays a limitation."""
    _, bag = lower("qbool q = false; qbool q2 = false; int t = measure(q); RX(t, q2);")
    assert bag.has_errors
    assert "not known at compile time" in bag.errors[0].message


def test_classical_name_is_a_renderable_angle(lower: LowerSource) -> None:
    """A classical constant has no runtime reassignment path yet, so a Name
    referencing it is just as renderable as the literal it was initialized
    with."""
    module, bag = lower("qbool q = false; float t = PI / 4; RX(t, q);")
    assert not bag.has_errors

    gate = module.body.ops[-1]
    assert isinstance(gate, GateOp)
    assert len(gate.params) == 1
    assert gate.params[0].name == "t"


def test_complex_angle_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; RX(0.5+0.3i, q);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_complex_phase_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("phase(1i);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_complex_classical_name_angle_is_rejected(lower: LowerSource) -> None:
    """Same rejection reached through a classical constant, not just a bare
    literal -- the complex check runs on the resolved value either way."""
    _, bag = lower("complex c = 1i; qbool q = false; RX(c, q);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_boolean_literal_angle_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; RX(true, q);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_boolean_classical_name_angle_is_rejected(lower: LowerSource) -> None:
    """Same rejection reached through a classical constant, not just a bare
    literal -- exercises the const_value Name resolution added alongside."""
    _, bag = lower("qbool q = false; bool flag = true; RX(flag, q);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_boolean_param_angle_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("param bool flag; qbool q = false; RX(flag, q);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_boolean_param_inside_arithmetic_angle_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("param bool flag; qbool q = false; RX(flag + 1, q);")
    assert bag.has_errors
    assert "cannot be used as an angle" in bag.errors[0].message


def test_phase_at_top_level_becomes_a_phase_op(lower: LowerSource) -> None:
    """A global phase is harmless outside a qif too, and meaningful if this
    circuit is later composed into something larger -- no reason to forbid it."""
    module, bag = lower("qbool q = false; phase(PI);")
    assert not bag.has_errors
    phase_op = module.body.ops[-1]
    assert isinstance(phase_op, PhaseOp)


def test_empty_probability_list_becomes_an_empty_ir_value(lower: LowerSource) -> None:
    """`[]` is the equal-superposition marker, kept as an empty list, not
    resolved into concrete probabilities here -- that is the codegen's job."""
    module, bag = lower("qint<2> a = [];")
    assert not bag.has_errors
    init = module.body.ops[0]
    assert isinstance(init, InitOp)
    assert init.value == []


def test_probability_list_reaches_the_ir_unnormalized(lower: LowerSource) -> None:
    """Lowering carries the source values through as-is; normalizing is a
    rendering concern, left to the codegen."""
    module, bag = lower("qint<2> a = [0.1, 0.1, 0.1, 0.1];")
    assert not bag.has_errors
    init = module.body.ops[0]
    assert isinstance(init, InitOp)
    assert init.value == [0.1, 0.1, 0.1, 0.1]


def test_empty_amplitude_list_becomes_an_empty_ir_value(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = {};")
    assert not bag.has_errors
    init = module.body.ops[0]
    assert isinstance(init, InitOp)
    assert init.value == []


def test_amplitude_list_reaches_the_ir_unnormalized_and_marked(lower: LowerSource) -> None:
    """Lowering carries the raw evaluated amplitudes through as-is (not yet
    normalized -- that is the codegen's job), and marks the op as an
    amplitude list rather than a probability list."""
    module, bag = lower("qint<1> a = {1/sqrt(2), 1i/sqrt(2)};")
    assert not bag.has_errors
    init = module.body.ops[0]
    assert isinstance(init, InitOp)
    assert init.is_amplitude is True
    assert init.value == [1 / 2**0.5, 1j / 2**0.5]


def test_amplitude_list_keeps_the_original_expressions(lower: LowerSource) -> None:
    """The original expressions are kept alongside the evaluated value so
    codegen can render an already-normalized amplitude symbolically instead
    of a folded decimal."""
    module, bag = lower("qint<1> a = {1/sqrt(2), 1i/sqrt(2)};")
    assert not bag.has_errors
    init = module.body.ops[0]
    assert isinstance(init, InitOp)
    assert init.value_expressions is not None
    assert len(init.value_expressions) == 2


def test_probability_list_is_not_marked_as_amplitude(lower: LowerSource) -> None:
    module, bag = lower("qint<1> a = [0.5, 0.5];")
    assert not bag.has_errors
    init = module.body.ops[0]
    assert isinstance(init, InitOp)
    assert init.is_amplitude is False


def test_qif_equality_condition_lowers_ctrl_state(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qbool out = false; qif(a == 2) { X(out); }")
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert qif.direct_qubits == [module.qubits[0]]
    assert qif.ctrl_state == 2
    assert qif.negated is False
    assert qif.ancilla is None
    assert [type(op).__name__ for op in qif.body.ops] == ["GateOp"]


def test_qif_and_chain_concatenates_ctrl_state(lower: LowerSource) -> None:
    """a==2 (bits 0,1) then flag (bit 1) concatenate to a 3-bit ctrl_state:
    0 | (1<<1) | (1<<2) == 6."""
    source = (
        "qint<2> a = 0; qbool flag = false; qbool out = false; "
        "qif(a == 2 && flag) { X(out); }"
    )
    module, bag = lower(source)
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert qif.ctrl_state == 6
    assert len(qif.direct_qubits) == 2


def test_qif_negated_single_qubit_flips_ctrl_state_without_ancilla(
    lower: LowerSource,
) -> None:
    module, bag = lower("qbool flag = false; qbool out = false; qif(!flag) { X(out); }")
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert qif.ctrl_state == 0
    assert qif.negated is False
    assert qif.ancilla is None


def test_qif_negated_multi_qubit_equality_needs_an_ancilla(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qbool out = false; qif(!(a == 2)) { X(out); }")
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert qif.ctrl_state == 2
    assert qif.negated is True
    assert qif.ancilla == QubitRef(name="_ancilla_0", size=1)


def test_qif_ancilla_name_collision_is_reported(lower: LowerSource) -> None:
    source = (
        "qint<2> a = 0; qbool _ancilla_0 = false; qbool out = false; "
        "qif(!(a == 2)) { X(out); }"
    )
    _, bag = lower(source)
    assert bag.has_errors
    assert "_ancilla_0" in bag.errors[0].message


def test_qif_phase_becomes_a_phase_op(lower: LowerSource) -> None:
    module, bag = lower(
        "qint<2> a = 0; qif(a == 2) { phase(1.5); }"
    )
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    (phase_op,) = qif.body.ops
    assert isinstance(phase_op, PhaseOp)
    assert const_value(phase_op.angle) == 1.5


def test_measure_inside_qif_is_rejected(lower: LowerSource) -> None:
    """A bare, unassigned measure() is already rejected by analysis before
    lowering runs -- the same as at the top level. The lowering's own
    qif-specific "measurement is not one" message is a defensive fallback for
    a path analysis does not currently let reach it."""
    _, bag = lower("qint<2> a = 0; qbool out = false; qif(a == 2) { measure(out); }")
    assert bag.has_errors
    assert "must be assigned" in bag.errors[0].message


def test_unimplemented_statement_inside_qif_body(lower: LowerSource) -> None:
    _, bag = lower(
        "qint<2> a = 0; qint<2> b = 0; qif(a == 2) { b += a; }"
    )
    assert bag.has_errors
    assert "compound assignment" in bag.errors[0].message
    assert "inside a qif body" in bag.errors[0].message


def test_a_process_definition_produces_no_operations(lower: LowerSource) -> None:
    """A template, not code: only the InitOp for `a` is left."""
    module, bag = lower("process f(qint x) { X(x); }\nqint<2> a = 0;\n")
    assert not bag.has_errors
    assert [type(op).__name__ for op in module.body.ops] == ["InitOp"]


def test_an_expanded_call_lowers_like_written_out_code(lower: LowerSource) -> None:
    module, bag = lower(
        "process add(qint x, qint y) { x += y; }\n"
        "qint<2> num1 = 1;\nqint<2> num2 = 2;\nadd(num1, num2);\n"
    )
    assert not bag.has_errors
    written_out, _ = lower(
        "qint<2> num1 = 1;\nqint<2> num2 = 2;\nnum1 += num2;\n"
    )
    assert [type(op).__name__ for op in module.body.ops] == [
        type(op).__name__ for op in written_out.body.ops
    ]


def test_an_unexpanded_process_call_is_an_internal_error(span: Span) -> None:
    """Lowering is reached with a resolved process call still in the tree only
    if the compiler itself skipped expansion, so the message says so rather
    than blaming the source. Built by hand because the real pipeline cannot
    produce this state -- which is the point."""
    process = ProcessDef(
        span=span,
        name="f",
        params=[ProcParam(span=span, name="x", declared_type=QBoolType())],
        body=Block(span=span),
    )
    callee = Name(span=span, name="f", resolved_symbol=process)
    ast = Program(
        span=span,
        statements=[ExprStatement(span=span, expr=Call(span=span, callee=callee))],
    )
    bag = DiagnosticBag()
    lower_to_ir(ast, bag)
    assert any("internal error" in diagnostic.message for diagnostic in bag.errors)


def test_an_undefined_call_is_not_blamed_on_the_compiler(lower: LowerSource) -> None:
    """It used to say "calling a process is not implemented yet" for any
    unknown callee, which was simply wrong for a typo."""
    _, bag = lower("qbool q = false;\nnosuchgate(q);\n")
    messages = [diagnostic.message for diagnostic in bag.errors]
    assert any("undefined name 'nosuchgate'" in message for message in messages)
    assert any("'nosuchgate' is not a gate" in message for message in messages)
    assert not any("limitation of the compiler" in message for message in messages)




def test_qif_not_equal_is_sugar_for_negated_equality(lower: LowerSource) -> None:
    """A wide `!=` standing alone goes through the clause-ancilla path, not
    the top-level one: it computes a==2 into its own ancilla, flips it, and
    controls the body directly by that ancilla -- no second, top-level
    ancilla needed. `!(a == 2)` reaches the same result the other way (see
    test_qif_negated_multi_qubit_equality_needs_an_ancilla)."""
    module, bag = lower("qint<2> a = 0; qbool out = false; qif(a != 2) { X(out); }")
    assert not bag.has_errors
    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert qif.negated is False
    assert qif.ancilla is None
    (clause_ancilla,) = qif.clause_ancillas
    assert clause_ancilla.ctrl_state == 2
    assert qif.direct_qubits == [QubitBit(ref=clause_ancilla.ancilla, index=0)]
    assert qif.ctrl_state == 1


def test_qif_equality_accepts_reversed_operands(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qbool out = false; qif(2 == a) { X(out); }")
    assert not bag.has_errors
    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert qif.ctrl_state == 2
    assert qif.negated is False


def test_qif_multiple_negated_clauses_each_get_their_own_ancilla(
    lower: LowerSource,
) -> None:
    source = (
        "qint<2> a = 0; qint<2> b = 0; qbool flag = false; qbool out = false; "
        "qif(!(a == 2) && !(b == 3) && flag) { X(out); }"
    )
    module, bag = lower(source)
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert len(qif.clause_ancillas) == 2
    assert qif.clause_ancillas[0].ctrl_state == 2
    assert qif.clause_ancillas[1].ctrl_state == 3
    # direct_qubits: both clause ancillas (bit=1 each) plus flag (bit=1).
    assert qif.ctrl_state == 0b111
    assert qif.negated is False
    assert qif.ancilla is None


def test_mvp_c_lowers_completely(lower: LowerSource, mvp_c_source: str) -> None:
    module, bag = lower(mvp_c_source)
    assert not bag.has_errors
    assert [type(op).__name__ for op in module.body.ops] == [
        "InitOp",
        "InitOp",
        "QIfOp",
        "MeasurementOp",
    ]


def _computation_ops(module: IRModule) -> list[Op]:
    """The module's operations without the ancilla declarations, which sit
    wherever a register is allocated rather than in one block."""
    return [op for op in module.body.ops if not isinstance(op, DeclareAncillaOp)]


def _declared_ancillas(module: IRModule) -> list[QubitRef]:
    return [
        op.ref for op in module.body.ops if isinstance(op, DeclareAncillaOp)
    ]


@pytest.mark.parametrize(
    "source",
    [
        "qint<2> a = 0; qint<2> b = 0; a += b;",
        "qint<3> a = 0; qint<2> b = 0; a += b;",
        "qint<2> a = 0; a += 3;",
        "qint<2> a = 0; qint<2> b = 0; qint<4> c = a * b;",
        "qint<2> a = 0; qint<2> b = 0; qint<2> c = 0; c += a * b;",
        "qint<2> a = 0; qint<2> b = 0; qint<3> c = 0; c += a * b;",
        "qint<2> a = 0; qint<2> b = 0; qint<4> c = 0; c += a * b;",
        "qint<2> a = 0; qint<2> b = 0; qint<5> c = 0; c += a * b;",
        "qint<2> a = 0; qint<2> b = 0; qint<4> c = 0; c += a * b; c += a * b;",
        "qint<2> a = 2; qbool t = false; qif(!(a == 2)) { X(t); }",
        "qint<2> a = 2; qint<2> b = 3; qbool t = false;"
        " qif(!(a == 2) && !(b == 3)) { X(t); }",
    ],
)
def test_every_ancilla_is_declared_exactly_once(
    lower: LowerSource, source: str
) -> None:
    """The invariant the declaration op exists for. `c += a * b` used to
    declare the product register twice -- once for the multiply, once again
    because the addend looked like an undeclared ancilla."""
    module, bag = lower(source)
    assert not bag.has_errors

    declared = _declared_ancillas(module)
    names = [ref.name for ref in declared]
    assert len(names) == len(set(names))

    used = {
        ref.name
        for op in _computation_ops(module)
        for ref in _referenced_refs(op)
        if ref.name.startswith("_ancilla_")
    }
    assert used == set(names)


def _referenced_refs(op: Op) -> list[QubitRef]:
    refs: list[QubitRef] = []
    for value in vars(op).values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, QubitRef):
                refs.append(item)
            elif isinstance(item, QubitBit | QubitSlice):
                refs.append(item.ref)
            elif isinstance(item, QIfClauseAncilla):
                refs.append(item.ancilla)
                refs += [
                    entry if isinstance(entry, QubitRef) else entry.ref
                    for entry in item.qubits
                ]
    return refs


@pytest.mark.parametrize(
    ("condition", "controls"),
    [
        # The repeated test says nothing the first one did not.
        ("a == 2 && a == 2", 2),
        ("a[0] && a[0]", 1),
        # Same qubit reached two ways: the register test already fixes a[0].
        ("a == 2 && a[0]", 2),
        ("a[0] && a == 2", 2),
        # The same negated test twice needs one ancilla, not two.
        ("!(a == 2) && !(a == 2)", 1),
    ],
)
def test_a_repeated_qif_test_is_dropped(
    lower: LowerSource, condition: str, controls: int
) -> None:
    """A qubit tested twice would reach the same append list twice, which
    Qiskit rejects as duplicate bit arguments when the generated file runs."""
    module, bag = lower(f"qint<2> a = 2; qbool t = false; qif({condition}) {{ X(t); }}")
    assert not bag.has_errors
    assert any("tests the same qubit more than once" in d.message for d in bag.warnings)

    (qif,) = [op for op in _computation_ops(module) if isinstance(op, QIfOp)]
    physical = [
        qubit
        for operand in qif.direct_qubits
        for qubit in (
            [(operand.name, index) for index in range(operand.size)]
            if isinstance(operand, QubitRef)
            else [(operand.ref.name, operand.index)]
        )
    ]
    assert len(physical) == len(set(physical))
    assert len(physical) == controls


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("a == 2 && a == 3", "can never be true"),
        ("a[0] && !(a[0])", "can never be true"),
        ("a == 2 && !(a == 2)", "can never be true"),
        ("!(a == 2 && a == 3)", "is always true"),
    ],
)
def test_a_qif_condition_no_state_can_satisfy_is_rejected(
    lower: LowerSource, condition: str, expected: str
) -> None:
    """Under a top-level negation the same finding flips meaning: the body is
    not conditional at all. Neither is representable as a ctrl_state, which
    names one basis pattern."""
    _, bag = lower(f"qint<2> a = 2; qbool t = false; qif({condition}) {{ X(t); }}")
    assert bag.has_errors
    assert expected in bag.errors[0].message


def test_a_dropped_clause_ancilla_leaves_no_declaration(lower: LowerSource) -> None:
    """The repeated negated test allocated an ancilla before it turned out to
    be a repetition; an ancilla nothing uses would still take a qubit."""
    module, bag = lower(
        "qint<2> a = 2; qbool t = false; qif(!(a == 2) && !(a == 2)) { X(t); }"
    )
    assert not bag.has_errors
    assert len(_declared_ancillas(module)) == 1


def test_a_circuit_a_simulator_cannot_hold_is_warned_about(
    lower: LowerSource,
) -> None:
    _, bag = lower("qint<31> a = 0;")
    assert not bag.has_errors
    assert "needs 31 qubits" in bag.warnings[0].message


def test_a_circuit_at_the_limit_is_not_warned_about(lower: LowerSource) -> None:
    _, bag = lower("qint<30> a = 0;")
    assert not bag.warnings


def test_the_qubit_count_is_the_whole_circuit_including_ancillas(
    lower: LowerSource,
) -> None:
    """Thirty declared qubits pass; the same program plus a `+=`, whose adder
    needs one scratch qubit of its own, does not. What a simulator cannot hold
    is the total, however the program divides it up."""
    _, bag = lower("qint<10> a = 0; qint<10> b = 0; qint<10> c = 0;")
    assert not bag.warnings

    _, bag = lower("qint<10> a = 0; qint<10> b = 0; qint<10> c = 0; a += b;")
    assert "needs 31 qubits" in bag.warnings[0].message


def test_augassign_equal_width_needs_no_padding(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; a += b;")
    assert not bag.has_errors

    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    assert arithmetic.target == module.qubits[0]
    assert arithmetic.addend == [module.qubits[1]]
    assert arithmetic.subtract is False
    assert arithmetic.encode_constant is None
    assert arithmetic.helper == QubitRef(name="_ancilla_0", size=1)


def test_augassign_subtract_sets_the_flag(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; a -= b;")
    assert not bag.has_errors
    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    assert arithmetic.subtract is True


def test_augassign_narrower_addend_gets_a_padding_ancilla(lower: LowerSource) -> None:
    module, bag = lower("qint<3> a = 0; qint<2> b = 0; a += b;")
    assert not bag.has_errors

    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    # _ancilla_0 is the adder's own helper, allocated before the padding.
    assert arithmetic.addend == [
        module.qubits[1],
        QubitRef(name="_ancilla_1", size=1),
    ]


def test_augassign_wider_addend_is_sliced_to_the_target_width(
    lower: LowerSource,
) -> None:
    module, bag = lower("qint<2> a = 0; qint<3> b = 0; a += b;")
    assert not bag.has_errors

    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    assert arithmetic.addend == [QubitSlice(ref=module.qubits[1], size=2)]


def test_augassign_constant_addend_encodes_into_a_fresh_ancilla(
    lower: LowerSource,
) -> None:
    module, bag = lower("qint<2> a = 0; a += 3;")
    assert not bag.has_errors

    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    assert arithmetic.encode_constant == 3
    (ancilla,) = arithmetic.addend
    # _ancilla_0 is the adder's own helper, allocated before the encoding ancilla.
    assert ancilla == QubitRef(name="_ancilla_1", size=2)


def test_augassign_constant_addend_wraps_modulo_the_target_width(
    lower: LowerSource,
) -> None:
    module, bag = lower("qint<2> a = 0; a += 6;")
    assert not bag.has_errors
    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    assert arithmetic.encode_constant == 2


def test_quantum_decl_multiply_becomes_a_multiply_op(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; qint<4> c = a * b;")
    assert not bag.has_errors

    multiply = module.body.ops[-1]
    assert isinstance(multiply, MultiplyOp)
    assert multiply.left == [module.qubits[0]]
    assert multiply.right == [module.qubits[1]]
    assert multiply.product == QubitRef(name="c", size=4)
    assert multiply.inverse is False
    assert module.qubits[-1] == QubitRef(name="c", size=4)


def test_quantum_decl_multiply_pads_the_narrower_operand(lower: LowerSource) -> None:
    module, bag = lower("qint<3> a = 0; qint<2> b = 0; qint<6> c = a * b;")
    assert not bag.has_errors

    multiply = module.body.ops[-1]
    assert isinstance(multiply, MultiplyOp)
    assert multiply.left == [module.qubits[0]]
    assert multiply.right == [
        module.qubits[1],
        QubitRef(name="_ancilla_0", size=1),
    ]


def test_multiply_accumulate_uses_a_temp_and_uncomputes_it(
    lower: LowerSource,
) -> None:
    """`d += a * b` must compute the product into a temp register, add it,
    then run the multiplier again to uncompute the temp -- `a`/`b` are never
    touched, so the same multiplier call cleanly reverses it."""
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; qint<3> d = 0; d += a * b;")
    assert not bag.has_errors

    ops = _computation_ops(module)[-3:]
    assert [type(op).__name__ for op in ops] == ["MultiplyOp", "ArithmeticOp", "MultiplyOp"]
    forward, arithmetic, backward = ops
    assert isinstance(forward, MultiplyOp)
    assert isinstance(arithmetic, ArithmeticOp)
    assert isinstance(backward, MultiplyOp)

    assert forward.inverse is False
    assert backward.inverse is True
    assert forward.product == backward.product
    assert forward.helper == backward.helper
    assert arithmetic.target == module.qubits[2]


def test_param_decl_registers_a_scalar_param(lower: LowerSource) -> None:
    module, bag = lower("param float theta;")
    assert not bag.has_errors
    (param,) = module.params
    assert param.name == "theta"
    assert param.kind == "scalar"
    assert param.size is None
    assert param.type_name == "float"


def test_param_array_decl_registers_its_size(lower: LowerSource) -> None:
    module, bag = lower("param int gamma[4];")
    assert not bag.has_errors
    (param,) = module.params
    assert param.kind == "array"
    assert param.size == 4
    assert param.type_name == "int"


def test_param_scalar_angle_is_accepted(lower: LowerSource) -> None:
    module, bag = lower("param float theta; qbool q = false; RX(theta, q);")
    assert not bag.has_errors
    gate = module.body.ops[-1]
    assert isinstance(gate, GateOp)
    assert len(gate.params) == 1


def test_param_expression_angle_is_accepted(lower: LowerSource) -> None:
    module, bag = lower(
        "param float theta; qbool q = false; RX(theta * 2 + PI / 4, q);"
    )
    assert not bag.has_errors
    gate = module.body.ops[-1]
    assert isinstance(gate, GateOp)
    assert len(gate.params) == 1


def test_param_array_indexed_angle_is_accepted(lower: LowerSource) -> None:
    module, bag = lower("param int gamma[4]; qbool q = false; RX(gamma[2], q);")
    assert not bag.has_errors
    gate = module.body.ops[-1]
    assert isinstance(gate, GateOp)
    assert len(gate.params) == 1


def test_param_with_an_unsupported_operator_is_reported(lower: LowerSource) -> None:
    _, bag = lower("param int gamma; qbool q = false; RX(gamma % 2, q);")
    assert bag.has_errors
    assert "not known at compile time" in bag.errors[0].message


def test_param_decl_inside_qif_body_is_a_top_level_only_error(
    lower: LowerSource,
) -> None:
    """A `param` is a whole-circuit, build-time object -- unlike AugAssign or
    a quantum declaration, no future qif-body work would ever make this
    legal, so it gets its own message, not '...not implemented yet'."""
    _, bag = lower("qint<2> a = 0; qif(a == 2) { param float x; }")
    assert bag.has_errors
    assert "only allowed at the top level of a program" in bag.errors[0].message
    assert "not implemented" not in bag.errors[0].message


# Every shape analysis rejects but lowering still walks into, because lowering
# runs after an error by design: each one used to raise AssertionError, killing
# the compiler on a program whose real error had already been reported.
REJECTED_SHAPES = [
    "qbool q = false;\nX();\n",
    "qbool q = false;\nX(q, q);\n",
    "qbool a = false;\nqbool b = false;\nCX(a);\n",
    "qbool a = false;\nqbool b = false;\nCX(a, b, a);\n",
    "qbool a = false;\nCCX(a, a);\n",
    "qbool q = false;\nRX(q);\n",
    "qbool q = false;\nRX(PI, q, q);\n",
    "phase();\n",
    "phase(PI, PI);\n",
    "qint<2> a = 0;\nint r = measure();\n",
    "qint<0> a = 0;\nqbool t = false;\nqif(!(a == 0) && a == 0) { X(t); }\n",
    "qint<0> a = 0;\nqbool t = false;\nqif(a == 0 && a == 0) { X(t); }\n",
    "qint<0> a = 0;\nqint<2> b = 0;\na += b;\n",
    "qint<0> a = 0;\nint r = measure(a);\n",
    "qint<2> a = 0;\nint r = measure(a, a);\n",
    "qint<2> a = 0;\nparam int g[2];\na += g[0];\n",
    "qint<2> a = 0;\nqint<2> b = 0;\na += b[0];\n",
    "qint<2> a = 0;\na += 1.5;\n",
    "qint<2> a = 0;\nqint<2> b = 0;\na[0] += b;\n",
    "qint<2> a = 0;\nqint<4> c = a * 3;\n",
    "qint<2> a = 0;\nqint<2> b = 0;\nqint<4> c = a * b[0];\n",
    "param complex t;\nqint<2> a = 0;\nRX(t, a[0]);\n",
    "qint<2> a = 0;\nqif(1) { X(a[0]); }\n",
    "qint<2> a = 0;\nqif(a + 1) { X(a[0]); }\n",
    "qint<2> a = 0;\nqif(1 == 2) { X(a[0]); }\n",
    "qint<2> a = 0;\nqif(a[0] == 1) { X(a[1]); }\n",
    "qint<1> a = {true, false};\n",
    "param float t;\nqint<1> a = {t, 0};\n",
]


@pytest.mark.parametrize("source", REJECTED_SHAPES)
def test_a_rejected_shape_does_not_crash_lowering(
    lower: LowerSource, source: str
) -> None:
    _, bag = lower(source)
    assert bag.has_errors
    assert not any("internal error" in error.message for error in bag.errors)


def test_a_call_with_the_wrong_argument_count_is_an_internal_error(span: Span) -> None:
    ast = Program(
        span=span,
        statements=[
            QuantumDecl(
                span=span,
                name="q",
                declared_type=QBoolType(),
                initializer=Literal(span=span, value=False),
            ),
            ExprStatement(
                span=span,
                expr=Call(
                    span=span,
                    callee=Name(span=span, name="X"),
                    args=[
                        Name(span=span, name="q"),
                        Name(span=span, name="q"),
                    ],
                ),
            ),
        ],
    )
    bag = DiagnosticBag()
    lower_to_ir(ast, bag)
    assert any("internal error" in error.message for error in bag.errors)
    assert any("2 argument(s) instead of 1" in error.message for error in bag.errors)


def test_an_unreachable_shape_with_no_error_is_an_internal_error(span: Span) -> None:
    """The other half of the guard: with an empty bag nothing rejected this, so
    analysis has a hole and the compiler says so instead of going quiet. Built
    by hand, since analysis does reject this shape."""
    ast = Program(
        span=span,
        statements=[
            QuantumDecl(
                span=span,
                name="a",
                declared_type=QIntType(size=2),
                initializer=Literal(span=span, value=0),
            ),
            AugAssign(
                span=span,
                target=Index(
                    span=span,
                    base=Name(span=span, name="a"),
                    index=Literal(span=span, value=0),
                ),
                op="+=",
                value=Name(span=span, name="a"),
            ),
        ],
    )
    bag = DiagnosticBag()
    lower_to_ir(ast, bag)
    assert any("internal error" in error.message for error in bag.errors)


def test_an_unrolled_loop_reaching_lowering_is_an_internal_error(span: Span) -> None:
    """A loop is replaced by its body or dropped, so lowering never meets one.
    Built by hand, since the pipeline cannot produce this state."""
    ast = Program(
        span=span,
        statements=[
            For(
                span=span,
                binding=LoopVarDecl(span=span, name="i", declared_type=IntType()),
                iterable=Call(
                    span=span,
                    callee=Name(span=span, name="range"),
                    args=[Literal(span=span, value=2)],
                ),
                body=Block(span=span),
            )
        ],
    )
    bag = DiagnosticBag()
    lower_to_ir(ast, bag)
    assert any("internal error" in error.message for error in bag.errors)
    assert not any("not implemented yet" in error.message for error in bag.errors)


def test_a_loop_lowers_to_its_unrolled_operations(lower: LowerSource) -> None:
    module, bag = lower("qint<3> a = 0;\nfor(int i in range(3)) { X(a[i]); }\n")
    assert not bag.has_errors
    gates = [op for op in module.body.ops if isinstance(op, GateOp)]
    assert len(gates) == 3
    assert [gate.name for gate in gates] == ["X", "X", "X"]


_IF_DECLS = "qint<2> a = 0; qbool f = false; int n = 5; bool t = true; "


def test_an_if_becomes_one_op_holding_its_branches(lower: LowerSource) -> None:
    module, bag = lower(f"{_IF_DECLS}if (t) {{ X(f); }} else {{ Y(f); }}")
    assert not bag.has_errors

    branch = module.body.ops[-1]
    assert isinstance(branch, ClassicalIfOp)
    assert [type(op).__name__ for op in branch.body.ops] == ["GateOp"]
    assert branch.orelse is not None
    assert [type(op).__name__ for op in branch.orelse.ops] == ["GateOp"]


def test_an_if_without_else_has_no_else_block(lower: LowerSource) -> None:
    module, bag = lower(f"{_IF_DECLS}if (t) {{ X(f); }}")
    assert not bag.has_errors
    branch = module.body.ops[-1]
    assert isinstance(branch, ClassicalIfOp)
    assert branch.orelse is None


def test_an_ancilla_from_a_branch_is_declared_at_the_top_level(
    lower: LowerSource,
) -> None:
    """The branch decides which gates run, not which registers exist: a
    register only added when one branch is taken would be missing from the
    circuit everywhere else."""
    module, bag = lower(f"{_IF_DECLS}qint<2> b = 1; if (t) {{ a += b; }}")
    assert not bag.has_errors

    assert any(isinstance(op, DeclareAncillaOp) for op in module.body.ops)
    branch = module.body.ops[-1]
    assert isinstance(branch, ClassicalIfOp)
    assert not any(isinstance(op, DeclareAncillaOp) for op in branch.body.ops)


def test_an_if_inside_a_qif_body_stays_a_branch(lower: LowerSource) -> None:
    module, bag = lower(f"{_IF_DECLS}qif(a == 2) {{ X(f); if (t) {{ Y(f); }} }}")
    assert not bag.has_errors

    qif = module.body.ops[-1]
    assert isinstance(qif, QIfOp)
    assert [type(op).__name__ for op in qif.body.ops] == ["GateOp", "ClassicalIfOp"]
