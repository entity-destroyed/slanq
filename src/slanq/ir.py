from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from slanq.ast_nodes import Expression, Span


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
    value: int | bool | list[float]


@dataclass(kw_only=True, eq=False)
class GateOp(Op):
    name: str
    targets: list[QubitOperand] = field(default_factory=list)
    # Kept as expressions, not numbers: `RX(PI/4, q)` must reach the generated
    # file as `np.pi / 4`, since 0.785... is unreadable.
    params: list[Expression] = field(default_factory=list)


@dataclass(kw_only=True, eq=False)
class MeasurementOp(Op):
    source: QubitOperand
    target: ClbitRef


@dataclass(kw_only=True, eq=False)
class PhaseOp(Op):
    """`phase()`. Applied to a circuit's own `global_phase` directly; inside a
    `qif` body's sub-circuit this becomes a relative phase once that
    sub-circuit is controlled."""

    angle: Expression


@dataclass(kw_only=True)
class IRBlock:
    ops: list[Op] = field(default_factory=list)


@dataclass(kw_only=True)
class QIfClauseAncilla:
    """One `&&`-chain clause whose own negation spans more than one qubit
    (`!(a == 2)` for a multi-qubit `a`, or `a != 2`): computed into its own
    ancilla by matching `ctrl_state` against `qubits`, then flipped. The
    flipped ancilla is that clause's one-bit contribution to the qif's
    overall match, folded into `QIfOp.direct_qubits`/`ctrl_state` alongside
    every other clause -- from there on it is just another required bit.
    Uncomputed the same way, in reverse, once the body has run."""

    qubits: list[QubitOperand]
    ctrl_state: int
    ancilla: QubitRef


@dataclass(kw_only=True, eq=False)
class QIfOp(Op):
    """A `qif`. `direct_qubits` entries are either a `QubitRef` (a whole
    register compared to a constant -- value semantics, no mirroring, same
    rule as `InitOp`) or a `QubitBit` (an explicitly indexed single bit, or a
    clause ancilla after its own negation -- mirrors like a gate operand, a
    no-op for an ancilla since it is always index 0). `ctrl_state`
    concatenates their required bits in that same order; `negated` flips the
    whole match.

    `clause_ancillas` lists the per-clause scratch qubits that had to be
    computed before `direct_qubits`/`ctrl_state` could be assembled --
    computed first, in this order, and uncomputed last, in reverse.

    `ancilla` is set exactly when `negated` is: negating the *whole* match
    once it spans more than one qubit needs one more scratch qubit (compute
    the positive match into it, flip it, control the body by it, flip and
    uncompute) -- a single bit's negation is absorbed into `ctrl_state`
    directly and needs no ancilla of its own."""

    direct_qubits: list[QubitOperand]
    ctrl_state: int
    clause_ancillas: list[QIfClauseAncilla]
    negated: bool
    ancilla: QubitRef | None
    body: IRBlock


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
    "PhaseOp",
    "QIfClauseAncilla",
    "QIfOp",
    "QubitBit",
    "QubitOperand",
    "QubitRef",
]
