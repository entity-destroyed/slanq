"""Gives every register the compiler allocated for itself real qubits, out of
one shared `_ancilla` register.

A qubit is handed on only where that cannot cost depth. Reusing a qubit last
touched by operation P for a new operation N makes N wait for P. If the program
already makes N wait for P -- some chain of operations, each sharing a qubit
with the next, leads from P to N -- that wait is there anyway, and the reuse
changes nothing about when anything can run. Otherwise the request gets a fresh
qubit, because two independent operations sharing a scratch qubit would have to
run one after the other.

The graph is built over the compiler's own register names, never over the
qubits they end up on. The two agree: every reuse only adds an edge the graph
already implied.

Every scratch register is back at |0> after its last use -- each one is
computed and uncomputed by the same lowering that allocated it -- so a qubit
needs nothing done to it between two owners.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slanq.ir import (
    DeclareAncillaOp,
    IRBlock,
    IRModule,
    Op,
    QIfOp,
    Qubit,
    QubitRef,
    operand_qubits,
    touched_qubits,
)


def allocate_ancillas(module: IRModule) -> None:
    scratch = {op.ref.name: op.ref for op in module.body.ops if isinstance(op, DeclareAncillaOp)}
    pool = _Pool(scratch)
    pool.allocate(module.body, inherited=[])

    module.ancilla_size = pool.size
    module.ancilla_qubits = pool.assigned
    # A register lowering allocated and then found it did not need is never
    # touched, so it gets no qubit and needs no declaration.
    module.body.ops = [
        op
        for op in module.body.ops
        if not (isinstance(op, DeclareAncillaOp) and op.ref.name not in pool.assigned)
    ]


@dataclass
class _Free:
    qubit: int
    # The operation in the current block that last used it, or None for a
    # qubit that was already free when the block began.
    last_user: int | None


@dataclass
class _Pool:
    scratch: dict[str, QubitRef]
    size: int = 0
    assigned: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def allocate(self, block: IRBlock, inherited: list[int]) -> set[int]:
        """Assigns the scratch registers that live in `block`, starting from
        the `inherited` qubits (oldest release first), and returns every qubit
        the block used, so the operation holding it can release them."""
        ops = [op for op in block.ops if not isinstance(op, DeclareAncillaOp)]
        first, last = self._lifetimes(ops)
        order = _Order()

        free = [_Free(qubit=qubit, last_user=None) for qubit in inherited]
        used: set[int] = set()
        for index, op in enumerate(ops):
            for name in first.get(index, []):
                waits = order.waits_for(_first_use(op, name))
                qubits = self._take(self.scratch[name].size, _ready(free, waits), free)
                self.assigned[name] = qubits
                used.update(qubits)

            # Two arms of a branch never run together, so a qubit one arm took
            # fresh is still |0> for the next.
            touched = touched_qubits(op)
            nested_used: set[int] = set()
            available = [entry.qubit for entry in _ready(free, order.waits_for(touched))]
            for nested in _nested_blocks(op):
                taken = self.allocate(nested, available)
                available += sorted(taken - set(available))
                nested_used |= taken
            if nested_used:
                free[:] = [entry for entry in free if entry.qubit not in nested_used]
                free += [_Free(qubit=qubit, last_user=index) for qubit in sorted(nested_used)]
                used |= nested_used

            for name in last.get(index, []):
                free += [_Free(qubit=qubit, last_user=index) for qubit in self.assigned[name]]
            order.add(touched)
        return used

    def _take(self, count: int, ready: list[_Free], free: list[_Free]) -> tuple[int, ...]:
        taken = ready[:count]
        free[:] = [entry for entry in free if entry not in taken]
        qubits = [entry.qubit for entry in taken]
        while len(qubits) < count:
            qubits.append(self.size)
            self.size += 1
        return tuple(qubits)

    def _lifetimes(self, ops: list[Op]) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
        """For each operation, the scratch registers that start and end living
        at it. A register lives in the outermost block that touches it, from
        its first touch to its last, a touch through a nested block included.
        Only one touched solely inside a single nested block of a single
        operation lives in that block instead, and is left to it; one an
        enclosing block already placed is never placed again. Touched in both
        arms of a branch, it lives around the branch: each arm would otherwise
        take it for its own and free it while the other still holds it."""
        touching: dict[str, list[int]] = {}
        nested_only: set[str] = set()
        for index, op in enumerate(ops):
            names = {
                name
                for name, _ in touched_qubits(op)
                if name in self.scratch and name not in self.assigned
            }
            direct = {name for name, _ in touched_qubits(op, nested=False)}
            arms = [_names_in(nested) for nested in _nested_blocks(op)]
            for name in sorted(names):
                touching.setdefault(name, []).append(index)
                if name not in direct and sum(name in arm for arm in arms) == 1:
                    nested_only.add(name)

        first: dict[int, list[str]] = {}
        last: dict[int, list[str]] = {}
        for name, indices in touching.items():
            if len(indices) == 1 and name in nested_only:
                continue
            first.setdefault(indices[0], []).append(name)
            last.setdefault(indices[-1], []).append(name)
        return first, last


def _nested_blocks(op: Op) -> list[IRBlock]:
    return [value for value in vars(op).values() if isinstance(value, IRBlock)]


def _names_in(block: IRBlock) -> set[str]:
    return {name for op in block.ops for name, _ in touched_qubits(op)}


def _ready(free: list[_Free], waits: int) -> list[_Free]:
    """The free qubits whose last user an operation already waits for."""
    return [entry for entry in free if entry.last_user is None or waits >> entry.last_user & 1]


def _first_use(op: Op, name: str) -> set[Qubit]:
    """The qubits of the first instruction that uses `name`, which is the one
    that would have to wait for the qubit's previous owner. For most operations
    that is the operation itself. A qif becomes several instructions: each
    negation ancilla is first computed from its own clause's qubits, the whole
    negation from the condition's, and only then comes the controlled body."""
    if isinstance(op, QIfOp):
        for clause in op.clause_ancillas:
            if clause.ancilla.name == name:
                return {qubit for operand in clause.qubits for qubit in operand_qubits(operand)}
        if op.ancilla is not None and op.ancilla.name == name:
            return {qubit for operand in op.direct_qubits for qubit in operand_qubits(operand)}
    return touched_qubits(op)


class _Order:
    """Which earlier operations of one block each one already has to wait for,
    as bit masks over their indices, built one operation at a time."""

    def __init__(self) -> None:
        self._last_toucher: dict[Qubit, int] = {}
        self._ancestors: list[int] = []

    def waits_for(self, qubits: set[Qubit]) -> int:
        mask = 0
        for qubit in qubits:
            previous = self._last_toucher.get(qubit)
            if previous is not None:
                mask |= 1 << previous | self._ancestors[previous]
        return mask

    def add(self, qubits: set[Qubit]) -> None:
        index = len(self._ancestors)
        self._ancestors.append(self.waits_for(qubits))
        for qubit in qubits:
            self._last_toucher[qubit] = index


__all__ = ["allocate_ancillas"]
