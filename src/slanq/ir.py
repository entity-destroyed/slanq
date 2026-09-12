from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from slanq.ast_nodes import Expression, Span


@dataclass(frozen=True, kw_only=True)
class QubitRef:
    """A whole quantum register. Frozen so analyses can use it as a dict/set key."""

    name: str
    size: int


@dataclass(frozen=True, kw_only=True)
class QubitBit:
    ref: QubitRef
    index: int


@dataclass(frozen=True, kw_only=True)
class ClbitRef:
    name: str
    size: int


@dataclass(frozen=True, kw_only=True)
class QubitSlice:
    """The first `size` physical qubits of `ref` -- value semantics, does NOT
    mirror, same rule as InitOp. Needed when a register is wider than an
    operation actually needs (e.g. the low bits of a wider addend for a
    `+=`)."""

    ref: QubitRef
    size: int


QubitOperand = QubitRef | QubitBit | QubitSlice


@dataclass(kw_only=True, eq=False)
class Op:
    """Base of all IR operations.

    `eq=False` keeps identity semantics, so ops stay hashable and analyses can
    key on them (`dict[Op, ...]`). Dataclass arguments are not inherited, so
    every subclass must repeat it.
    """

    span: Span


@dataclass(kw_only=True, eq=False)
class DeclareAncillaOp(Op):
    """Brings a compiler-allocated register into the circuit. Emitted once,
    where the register is allocated, so no later phase has to work out from an
    operand whether it has been declared yet -- guessing that from the operand
    is how the same register came to be declared twice."""

    ref: QubitRef


@dataclass(kw_only=True, eq=False)
class InitOp(Op):
    target: QubitRef
    value: int | bool | list[float] | list[complex | float]
    # True for a `{}` amplitude list: `value` holds raw, not-yet-normalized
    # amplitudes (int/float/complex), not probabilities -- codegen divides by
    # the L2 norm instead of taking a square root.
    is_amplitude: bool = False
    # Only set when is_amplitude is True: each element's original Slanq
    # expression, kept alongside the evaluated `value` so codegen can render
    # an already-normalized amplitude symbolically (e.g. `1 / math.sqrt(2)`)
    # instead of a decimal approximation -- the same "never fold a
    # compile-time expression" rule angle arguments already follow.
    value_expressions: list[Expression] | None = None


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
class ResetOp(Op):
    target: QubitOperand


@dataclass(kw_only=True, eq=False)
class DeclareRealtimeOp(Op):
    """A classical variable on the processor beside the QPU, started from a
    build-time value. A measurement's result needs no such op: its classical
    register is already real-time."""

    name: str
    size: int
    value: Expression
    # `true`/`false` reach Qiskit as 1/0: every real-time value is a Uint, and
    # a truth value is the one-bit case of that.
    is_bool: bool = False


@dataclass(kw_only=True, eq=False)
class RealtimeIfOp(Op):
    """A branch taken while the circuit runs, on a real-time classical value,
    so it becomes a Qiskit `if_test` block."""

    condition: Expression
    body: IRBlock
    orelse: IRBlock | None = None


@dataclass(kw_only=True, eq=False)
class ClassicalIfOp(Op):
    """A branch taken while the circuit is being built, so it becomes a Python
    `if` in the generated file and the gates of the branch not taken are never
    appended."""

    condition: Expression
    body: IRBlock
    orelse: IRBlock | None = None


@dataclass(kw_only=True, eq=False)
class PhaseOp(Op):
    """`phase()`. Applied to a circuit's own `global_phase` directly; inside a
    `qif` body's sub-circuit this becomes a relative phase once that
    sub-circuit is controlled."""

    angle: Expression


@dataclass(kw_only=True, eq=False)
class ArithmeticOp(Op):
    """`target += addend` or `target -= addend` (`subtract`). `addend` is
    always target.size physical qubits long, concatenated -- may be the real
    variable (possibly sliced via QubitSlice), plus a fresh 0-ancilla if the
    real addend is narrower. `encode_constant` is set when the addend is a
    compile-time constant: then `addend` is a fresh ancilla allocated only
    for this purpose, which codegen encodes with X-gates before the add and
    decodes with the same gates afterward. `helper` is the adder's own
    self-restoring scratch qubit, fresh every time."""

    target: QubitRef
    addend: list[QubitOperand]
    helper: QubitRef
    subtract: bool
    encode_constant: int | None


@dataclass(kw_only=True, eq=False)
class MultiplyOp(Op):
    """`product := left * right` (or, with `inverse`, the same multiplier run
    backwards to uncompute a temporary `product`). `left`/`right` are padded
    to equal width with a 0-ancilla if their sizes differed -- also
    discardable without uncompute, since the multiplier's inputs are left
    unchanged. `product` is sized to the caller's need (silently truncated
    mod 2^size if smaller than `2*width`). `helper` is the multiplier's own
    scratch qubit, fresh every time."""

    left: list[QubitOperand]
    right: list[QubitOperand]
    product: QubitRef
    helper: QubitRef
    inverse: bool


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


@dataclass(frozen=True, kw_only=True)
class ParamInfo:
    """A declared `param`/`param[]`, tracked so the codegen knows what
    `Parameter` objects and `build_bound_circuit` signature to emit. `size`
    is the element count for an array, `None` for a scalar."""

    name: str
    kind: Literal["scalar", "array"]
    size: int | None
    type_name: Literal["int", "float", "bool"]


@dataclass(kw_only=True)
class IRModule:
    qubits: list[QubitRef] = field(default_factory=list)
    clbits: list[ClbitRef] = field(default_factory=list)
    params: list[ParamInfo] = field(default_factory=list)
    body: IRBlock = field(default_factory=IRBlock)


__all__ = [
    "ArithmeticOp",
    "ClassicalIfOp",
    "ClbitRef",
    "DeclareAncillaOp",
    "DeclareRealtimeOp",
    "GateOp",
    "IRBlock",
    "IRModule",
    "InitOp",
    "MeasurementOp",
    "MultiplyOp",
    "Op",
    "ParamInfo",
    "PhaseOp",
    "QIfClauseAncilla",
    "QIfOp",
    "QubitBit",
    "QubitOperand",
    "QubitRef",
    "QubitSlice",
    "RealtimeIfOp",
    "ResetOp",
]
