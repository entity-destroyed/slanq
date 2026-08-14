from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from slanq.ast_nodes import Span
from slanq.ir import (
    ClbitRef,
    GateOp,
    InitOp,
    IRBlock,
    IRModule,
    MeasurementOp,
    QubitBit,
    QubitRef,
)


def test_qubit_ref_is_frozen() -> None:
    ref = QubitRef(name="q", size=1)
    with pytest.raises(FrozenInstanceError):
        ref.name = "other"


def test_refs_compare_by_value_and_hash() -> None:
    a = QubitRef(name="q", size=1)
    b = QubitRef(name="q", size=1)
    assert a == b
    assert len({a, b}) == 1


def test_qubit_bit_compares_by_value() -> None:
    ref = QubitRef(name="q", size=2)
    assert QubitBit(ref=ref, index=0) == QubitBit(ref=ref, index=0)
    assert QubitBit(ref=ref, index=0) != QubitBit(ref=ref, index=1)


def test_origin_defaults_to_user() -> None:
    assert QubitRef(name="q", size=1).origin == "user"
    assert QubitRef(name="$ancilla_0", size=1, origin="ancilla").origin == "ancilla"


def test_ops_compare_by_identity(span: Span) -> None:
    ref = QubitRef(name="q", size=1)
    first = GateOp(span=span, name="H", targets=[QubitBit(ref=ref, index=0)])
    second = GateOp(span=span, name="H", targets=[QubitBit(ref=ref, index=0)])
    assert first != second
    assert first == first


def test_ops_are_hashable_so_analyses_can_key_on_them(span: Span) -> None:
    ref = QubitRef(name="q", size=1)
    op = GateOp(span=span, name="H", targets=[QubitBit(ref=ref, index=0)])
    live: dict[GateOp, set[QubitRef]] = {op: {ref}}
    assert live[op] == {ref}


def test_ops_are_mutable_so_passes_can_rewrite_operands(span: Span) -> None:
    original = QubitRef(name="q", size=1)
    replacement = QubitRef(name="$ancilla_0", size=1, origin="ancilla")
    op = GateOp(span=span, name="H", targets=[QubitBit(ref=original, index=0)])

    op.targets[0] = QubitBit(ref=replacement, index=0)

    assert op.targets == [QubitBit(ref=replacement, index=0)]


def test_op_default_lists_are_independent(span: Span) -> None:
    first = GateOp(span=span, name="H")
    second = GateOp(span=span, name="X")
    first.targets.append(QubitBit(ref=QubitRef(name="q", size=1), index=0))
    assert len(first.targets) == 1
    assert len(second.targets) == 0


def test_module_gets_a_fresh_body_per_instance(span: Span) -> None:
    first = IRModule()
    second = IRModule()
    first.body.ops.append(InitOp(span=span, target=QubitRef(name="q", size=1), value=False))
    assert len(first.body.ops) == 1
    assert len(second.body.ops) == 0


def test_mvp_a_ir_shape(mvp_a_ir: IRModule) -> None:
    (qubit,) = mvp_a_ir.qubits
    (clbit,) = mvp_a_ir.clbits
    assert qubit == QubitRef(name="q", size=1)
    assert clbit == ClbitRef(name="result", size=1)

    init, gate, measurement = mvp_a_ir.body.ops
    assert isinstance(init, InitOp)
    assert init.value is False
    assert isinstance(gate, GateOp)
    assert gate.name == "H"
    assert gate.targets == [QubitBit(ref=qubit, index=0)]
    assert isinstance(measurement, MeasurementOp)
    assert measurement.source == qubit
    assert measurement.target == clbit


def test_ir_block_default_ops_are_independent() -> None:
    assert IRBlock().ops == []
    assert IRBlock().ops is not IRBlock().ops
