from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import Program
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
    module, bag = lower("qbool q = false; RX(q, 90);")
    assert not bag.has_errors

    gate = module.body.ops[1]
    assert isinstance(gate, GateOp)
    assert gate.params == [90.0]
    assert gate.targets == [QubitBit(ref=module.qubits[0], index=0)]


def test_classical_constant_produces_no_ir(lower: LowerSource) -> None:
    module, bag = lower("int x = 42;")
    assert not bag.has_errors
    assert module.clbits == []
    assert module.body.ops == []


def test_bare_measure_statement_is_rejected(lower: LowerSource) -> None:
    _, bag = lower("qbool q = false; measure(q);")
    assert bag.has_errors
    assert "measure()" in bag.errors[0].message


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
