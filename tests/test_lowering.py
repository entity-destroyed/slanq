from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import Program
from slanq.builtin import const_value
from slanq.diagnostics import DiagnosticBag
from slanq.ir import (
    ArithmeticOp,
    ClbitRef,
    GateOp,
    InitOp,
    IRModule,
    MeasurementOp,
    MultiplyOp,
    PhaseOp,
    QIfOp,
    QubitBit,
    QubitRef,
    QubitSlice,
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
    _, bag = lower("qbool q = false; if(1) { X(q); }")
    assert "if statement" in bag.errors[0].message


def test_unimplemented_message_blames_the_compiler(lower: LowerSource) -> None:
    """A quantum variable initialized from another variable: `const_value`
    never resolves a `Name`, so this stays a limitation, not a crash."""
    _, bag = lower("int t = 5; qint<2> b = t;")
    assert bag.has_errors
    assert "limitation of the compiler" in bag.errors[0].message


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
    _, bag = lower("qbool q = false; float t = 1.0; RX(t, q);")
    assert bag.has_errors
    assert "not known at compile time" in bag.errors[0].message


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
    assert qif.ancilla == QubitRef(name="_ancilla_0", size=1, origin="ancilla")


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


def test_augassign_equal_width_needs_no_padding(lower: LowerSource) -> None:
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; a += b;")
    assert not bag.has_errors

    arithmetic = module.body.ops[-1]
    assert isinstance(arithmetic, ArithmeticOp)
    assert arithmetic.target == module.qubits[0]
    assert arithmetic.addend == [module.qubits[1]]
    assert arithmetic.subtract is False
    assert arithmetic.encode_constant is None
    assert arithmetic.helper == QubitRef(name="_ancilla_0", size=1, origin="ancilla")


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
        QubitRef(name="_ancilla_1", size=1, origin="ancilla"),
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
    assert ancilla == QubitRef(name="_ancilla_1", size=2, origin="ancilla")


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
        QubitRef(name="_ancilla_0", size=1, origin="ancilla"),
    ]


def test_multiply_accumulate_uses_a_temp_and_uncomputes_it(
    lower: LowerSource,
) -> None:
    """`d += a * b` must compute the product into a temp register, add it,
    then run the multiplier again to uncompute the temp -- `a`/`b` are never
    touched, so the same multiplier call cleanly reverses it."""
    module, bag = lower("qint<2> a = 0; qint<2> b = 0; qint<3> d = 0; d += a * b;")
    assert not bag.has_errors

    ops = module.body.ops[-3:]
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
