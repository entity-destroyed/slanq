"""Which of the program's qubits have been tangled up with which, and what was
done to each of them, in order.

A node is one qubit of a variable. It keeps the variable it belongs to, its bit,
and every operation that touched it, numbered in program order and marked as a
read or a write. An edge joins two qubits of different variables that one
operation touched together -- the only way two of them can become entangled. It
is a safe overestimate: an edge does not mean they are entangled, its absence
means they cannot be. Measuring or resetting a whole variable leaves it in a
basis state, entangled with nothing, so its edges go.

A control structure (`qif`, `rt if`, `rt while`) is one operation here, touching
everything its body touches. The compiler's own scratch registers are not
variables and get no nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from slanq.ir import (
    ArithmeticOp,
    InitOp,
    MeasurementOp,
    MultiplyOp,
    Op,
    QIfOp,
    Qubit,
    QubitOperand,
    QubitRef,
    RealtimeIfOp,
    RealtimeWhileOp,
    RenameOp,
    ResetOp,
    operand_qubits,
    touched_qubits,
)

Role = Literal["read", "write"]


@dataclass(frozen=True)
class Touch:
    seq: int
    op: Op
    role: Role


@dataclass
class QubitNode:
    register: str
    index: int
    history: list[Touch] = field(default_factory=list)


@dataclass(frozen=True)
class Reuse:
    """A dead variable whose qubits can go back to the pool. `last_use` is the
    number of the operation that last touched it, or None if nothing ever did;
    `history` is what has to be run backwards to bring it to |0>, empty when it
    is there already."""

    last_use: int | None
    history: tuple[Op, ...]

    @property
    def needs_uncompute(self) -> bool:
        return bool(self.history)


# Operations that cannot be run backwards, whose effect depends on what
# happens while the circuit runs, or that hand the qubits to another name.
_IRREVERSIBLE = (MeasurementOp, ResetOp, RealtimeIfOp, RealtimeWhileOp, RenameOp)


class EntanglementGraph:
    def __init__(self, registers: list[QubitRef]) -> None:
        self.registers = {register.name: register for register in registers}
        self.nodes: dict[Qubit, QubitNode] = {
            (register.name, index): QubitNode(register=register.name, index=index)
            for register in registers
            for index in range(register.size)
        }
        self.edges: dict[Qubit, set[Qubit]] = {qubit: set() for qubit in self.nodes}

    @classmethod
    def build(cls, ops: list[Op], registers: list[QubitRef]) -> EntanglementGraph:
        graph = cls(registers)
        for seq, op in enumerate(ops):
            graph.add(seq, op)
        return graph

    def add(self, seq: int, op: Op) -> None:
        if _emits_nothing(op):
            return
        reads = _read_qubits(op)
        touched = [qubit for qubit in touched_qubits(op) if qubit in self.nodes]
        for qubit in touched:
            role: Role = "read" if qubit in reads else "write"
            self.nodes[qubit].history.append(Touch(seq=seq, op=op, role=role))
        for qubit in touched:
            for other in touched:
                if qubit[0] != other[0]:
                    self.edges[qubit].add(other)

        whole = _whole_register(op)
        if whole is not None and whole.name in self.registers:
            for qubit in self._qubits_of(whole.name):
                for other in self.edges[qubit]:
                    self.edges[other].discard(qubit)
                self.edges[qubit].clear()

    def variable_edges(self) -> set[frozenset[str]]:
        """The same graph seen per variable: which variables may be entangled."""
        return {
            frozenset((qubit[0], other[0]))
            for qubit, others in self.edges.items()
            for other in others
        }

    def reusable(self, name: str) -> Reuse | None:
        """Whether the variable's qubits can join the pool once it is dead,
        without anything the program measures changing.

        Free outright if nothing ever touched it, or if the last thing to was a
        reset of the whole variable. Otherwise only if, since its last whole
        reset, every operation on it touched its own qubits and nothing else,
        and each can be run backwards: such a variable was never entangled with
        anything, so undoing it is invisible to the rest of the program."""
        own = set(self._qubits_of(name))
        by_seq = {touch.seq: touch for qubit in own for touch in self.nodes[qubit].history}
        touches = [by_seq[seq] for seq in sorted(by_seq)]
        if not touches:
            return Reuse(last_use=None, history=())

        reset = _last_whole_reset(touches, self.registers[name])
        since = [touch for touch in touches if touch.seq > reset]
        if not since:
            return Reuse(last_use=touches[-1].seq, history=())
        for touch in since:
            if isinstance(touch.op, _IRREVERSIBLE) or not touched_qubits(touch.op) <= own:
                return None
        return Reuse(last_use=since[-1].seq, history=tuple(touch.op for touch in since))

    def _qubits_of(self, name: str) -> list[Qubit]:
        return [(name, index) for index in range(self.registers[name].size)]


def _last_whole_reset(touches: list[Touch], register: QubitRef) -> int:
    resets = [
        touch.seq
        for touch in touches
        if isinstance(touch.op, ResetOp) and _whole_register(touch.op) == register
    ]
    return max(resets, default=-1)


def _emits_nothing(op: Op) -> bool:
    """A variable declared as 0 starts where every qubit starts."""
    return isinstance(op, InitOp) and not isinstance(op.value, list) and not op.value


def _whole_register(op: Op) -> QubitRef | None:
    """The variable a measurement or reset leaves wholly in a basis state. A
    one-qubit variable reaches it as its single bit, which is all of it."""
    if isinstance(op, MeasurementOp):
        operand = op.source
    elif isinstance(op, ResetOp):
        operand = op.target
    else:
        return None
    register = operand if isinstance(operand, QubitRef) else operand.ref
    return register if len(operand_qubits(operand)) == register.size else None


def _read_qubits(op: Op) -> set[Qubit]:
    """The qubits an operation only reads, which the node records; no rule here
    turns on it, because a read is enough to tangle the two qubits. A gate's
    control counts as written for the same reason: its own value survives, but
    entanglement is symmetric, so afterwards its state can no longer be written
    down on its own."""
    operands: list[QubitOperand] = []
    if isinstance(op, ArithmeticOp):
        operands = list(op.addend)
    elif isinstance(op, MultiplyOp):
        operands = [*op.left, *op.right]
    elif isinstance(op, QIfOp):
        operands = list(op.direct_qubits)
        for clause in op.clause_ancillas:
            operands += clause.qubits
    elif isinstance(op, MeasurementOp):
        operands = [op.source]
    return {qubit for operand in operands for qubit in operand_qubits(operand)}


__all__ = ["EntanglementGraph", "QubitNode", "Reuse", "Touch"]
