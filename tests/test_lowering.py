from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import Program
from slanq.builtin import const_value
from slanq.diagnostics import DiagnosticBag
from slanq.ir import ClbitRef, GateOp, InitOp, IRModule, MeasurementOp, QubitBit, QubitRef
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
    generated circuit simply never performed the addition."""
    _, bag = lower("qint<2> a = 0; qint<2> b = 1; a += b;")
    assert bag.has_errors
    assert "not implemented yet" in bag.errors[0].message


def test_unimplemented_statement_names_the_construct(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; if(1) { X(q); }")
    assert "if statement" in bag.errors[0].message


def test_unimplemented_message_blames_the_compiler(lower: LowerSource) -> None:
    _, bag = lower("qint<2> b = [];")
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


def test_phase_is_reported_as_a_limitation(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; phase(PI);")
    assert bag.has_errors
    assert "phase gate is not implemented" in bag.errors[0].message
