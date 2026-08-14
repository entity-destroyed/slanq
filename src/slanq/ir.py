from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from slanq.ast_nodes import Span


@dataclass(frozen=True, kw_only=True)
class QubitRef:
    """A whole quantum register. Frozen so analyses can use it as a dict/set key."""

    name: str
    size: int
    origin: Literal["user", "ancilla"] = "user"


@dataclass(frozen=True, kw_only=True)
class QubitBit:
    ref: QubitRef
    index: int


@dataclass(frozen=True, kw_only=True)
class ClbitRef:
    name: str
    size: int


QubitOperand = QubitRef | QubitBit


@dataclass(kw_only=True, eq=False)
class Op:
    """Base of all IR operations.

    `eq=False` keeps identity semantics, so ops stay hashable and analyses can
    key on them (`dict[Op, ...]`). Dataclass arguments are not inherited, so
    every subclass must repeat it.
    """

    span: Span


@dataclass(kw_only=True, eq=False)
class InitOp(Op):
    target: QubitRef
    value: int | bool


@dataclass(kw_only=True, eq=False)
class GateOp(Op):
    name: str
    targets: list[QubitOperand] = field(default_factory=list)
    params: list[float] = field(default_factory=list)


@dataclass(kw_only=True, eq=False)
class MeasurementOp(Op):
    source: QubitOperand
    target: ClbitRef


@dataclass(kw_only=True)
class IRBlock:
    ops: list[Op] = field(default_factory=list)


@dataclass(kw_only=True)
class IRModule:
    qubits: list[QubitRef] = field(default_factory=list)
    clbits: list[ClbitRef] = field(default_factory=list)
    body: IRBlock = field(default_factory=IRBlock)


__all__ = [
    "ClbitRef",
    "GateOp",
    "IRBlock",
    "IRModule",
    "InitOp",
    "MeasurementOp",
    "Op",
    "QubitBit",
    "QubitOperand",
    "QubitRef",
]
