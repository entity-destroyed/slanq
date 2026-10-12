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

A dead variable's qubits join the pool too, behind the compiler's own, when the
entanglement graph shows that bringing them back to |0> cannot be observed:
the variable was never entangled with anything, so running its own history
backwards undoes it and nothing else. The uncomputation goes right after the
variable's last use, but only if some later register really is given one of its
qubits: a variable nobody needs the qubits of is left alone, and pays nothing.

No chain of operations can lead on from such a variable's last use -- nothing
touches it again -- so the rule above would never hand its qubits on. They go
by the timing the rule protects instead: every operation has a layer, the
earliest step it could run in, and a new operation may take them only if the
last use, and the uncomputation after it, are over before that operation's own
layer comes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slanq.entanglement import EntanglementGraph, Reuse
from slanq.ir import (
    DeclareAncillaOp,
    IRBlock,
    IRModule,
    Op,
    QIfOp,
    Qubit,
    QubitRef,
    UncomputeOp,
    operand_qubits,
    touched_qubits,
)

ANCILLA_REGISTER = "_ancilla"


def allocate_ancillas(module: IRModule) -> None:
    scratch = {op.ref.name: op.ref for op in module.body.ops if isinstance(op, DeclareAncillaOp)}
    ops = _operations(module.body)
    graph = EntanglementGraph.build(ops, module.qubits)
    dead = {
        register: reuse
        for register in module.qubits
        if (reuse := graph.reusable(register.name)) is not None
    }

    pool = _Pool(scratch, dead)
    pool.allocate(module.body, inherited=[], top_level=True)

    module.ancilla_size = pool.size
    module.ancilla_qubits = pool.assigned
    for register in pool.uncomputed:
        reuse = dead[register]
        assert reuse.last_use is not None
        last_use = ops[reuse.last_use]
        uncompute = UncomputeOp(span=last_use.span, target=register, history=list(reuse.history))
        module.body.ops.insert(module.body.ops.index(last_use) + 1, uncompute)


@dataclass
class _Free:
    qubit: Qubit
    # The operation in the current block that last used it, or None for a
    # qubit that was already free when the block began.
    last_user: int | None
    # For a dead variable's qubit, how many layers still have to run after its
    # last use before it is |0>: the uncomputation's own depth, none after a
    # reset. None for the compiler's own qubits, which go by the chain of
    # operations.
    cost: int | None = None


@dataclass
class _Pool:
    scratch: dict[str, QubitRef]
    dead: dict[QubitRef, Reuse]
    size: int = 0
    assigned: dict[str, tuple[Qubit, ...]] = field(default_factory=dict)
    uncomputed: list[QubitRef] = field(default_factory=list)
    # The qubits of dead variables still waiting to be uncomputed, and whose.
    _pending: dict[Qubit, QubitRef] = field(default_factory=dict)

    def allocate(
        self, block: IRBlock, inherited: list[Qubit], top_level: bool = False
    ) -> set[Qubit]:
        """Assigns the scratch registers that live in `block`, starting from
        the `inherited` qubits (oldest release first), and returns every qubit
        the block used, so the operation holding it can release them."""
        ops = _operations(block)
        first, last = self._lifetimes(ops)
        order = _Order()

        free = [_Free(qubit=qubit, last_user=None) for qubit in inherited]
        if top_level:
            free += self._dead_from_the_start()
        used: set[Qubit] = set()
        for index, op in enumerate(ops):
            for name in first.get(index, []):
                ready = self._ready(free, order, _first_use(op, name))
                qubits = self._take(self.scratch[name].size, ready, free)
                self.assigned[name] = qubits
                used.update(qubits)

            # Two arms of a branch never run together, so a qubit one arm took
            # fresh is still |0> for the next.
            touched = touched_qubits(op)
            nested_used: set[Qubit] = set()
            available = [entry.qubit for entry in self._ready(free, order, touched)]
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
            if top_level:
                free += self._dying_at(index)
            order.add(touched)
        return used

    def _dead_from_the_start(self) -> list[_Free]:
        """Variables nothing ever touches: their qubits are |0> and wait for
        nothing."""
        return [
            _Free(qubit=(register.name, bit), last_user=None)
            for register, reuse in self.dead.items()
            if reuse.last_use is None
            for bit in range(register.size)
        ]

    def _dying_at(self, index: int) -> list[_Free]:
        entries = []
        for register, reuse in self.dead.items():
            if reuse.last_use != index:
                continue
            for bit in range(register.size):
                qubit = (register.name, bit)
                if reuse.needs_uncompute:
                    self._pending[qubit] = register
                cost = _layer_depth(reuse.history)
                entries.append(_Free(qubit=qubit, last_user=index, cost=cost))
        return entries

    def _ready(self, free: list[_Free], order: _Order, qubits: set[Qubit]) -> list[_Free]:
        """The free qubits an operation on `qubits` can take without waiting
        for anything it did not wait for already, the ones already at |0> first:
        a dead variable's qubits cost an uncomputation."""
        waits = order.waits_for(qubits)
        layer = order.layer_of(qubits)

        def usable(entry: _Free) -> bool:
            if entry.last_user is None:
                return True
            if entry.cost is None:
                return bool(waits >> entry.last_user & 1)
            return order.layer(entry.last_user) + entry.cost < layer

        ready = [entry for entry in free if usable(entry)]
        return sorted(ready, key=lambda entry: entry.qubit in self._pending)

    def _take(self, count: int, ready: list[_Free], free: list[_Free]) -> tuple[Qubit, ...]:
        """`count` qubits for one register: the longest-free suitable ones, then
        fresh ones from `_ancilla` for whatever is still missing. Taking one
        qubit of a dead variable commits the whole variable to being
        uncomputed, so the rest of it is no longer free for the taking."""
        taken = ready[:count]
        free[:] = [entry for entry in free if entry not in taken]
        qubits = [entry.qubit for entry in taken]
        for qubit in qubits:
            owner = self._pending.get(qubit)
            if owner is not None:
                self.uncomputed.append(owner)
                for bit in range(owner.size):
                    del self._pending[(owner.name, bit)]
        while len(qubits) < count:
            qubits.append((ANCILLA_REGISTER, self.size))
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


def _layer_depth(history: tuple[Op, ...]) -> int:
    """How many layers the uncomputation of `history` takes. Operations on
    different qubits of the one variable run side by side, so this is not
    simply how many there are."""
    order = _Order()
    depth = 0
    for op in history:
        qubits = touched_qubits(op)
        depth = max(depth, order.layer_of(qubits))
        order.add(qubits)
    return depth


def _operations(block: IRBlock) -> list[Op]:
    """The block's operations, numbered as the allocation and the
    entanglement graph both count them."""
    return [op for op in block.ops if not isinstance(op, DeclareAncillaOp)]


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
        self._layers: list[int] = []

    def layer(self, index: int) -> int:
        return self._layers[index]

    def layer_of(self, qubits: set[Qubit]) -> int:
        """The earliest step an operation on `qubits` could run in, counting
        whole operations: one after the latest of those it has to wait for."""
        previous = [self._last_toucher[qubit] for qubit in qubits if qubit in self._last_toucher]
        return 1 + max((self._layers[index] for index in previous), default=0)

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
        self._layers.append(self.layer_of(qubits))
        for qubit in qubits:
            self._last_toucher[qubit] = index


__all__ = ["allocate_ancillas"]
