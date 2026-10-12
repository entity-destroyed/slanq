from __future__ import annotations

from typing import Any

import pytest
from qiskit import QuantumCircuit

from slanq import ancilla_pool
from slanq.ast_nodes import Literal, Span
from slanq.compiler import compile_source
from slanq.ir import (
    ArithmeticOp,
    DeclareAncillaOp,
    IRBlock,
    IRModule,
    QIfOp,
    QubitBit,
    QubitRef,
    RealtimeIfOp,
    UncomputeOp,
)


def _module(source: str) -> IRModule:
    result = compile_source(source, source_name="test.slanq", stop_after="ir")
    assert not result.diagnostics.has_errors, [str(d) for d in result.diagnostics]
    assert result.ir is not None
    return result.ir


def _circuit(source: str) -> QuantumCircuit:
    result = compile_source(source, source_name="test.slanq")
    assert not result.diagnostics.has_errors, [str(d) for d in result.diagnostics]
    assert result.qiskit_source is not None
    namespace: dict[str, Any] = {}
    exec(result.qiskit_source, namespace)  # noqa: S102
    circuit = namespace["build_circuit"]()
    str(circuit)
    return circuit


def _pool_qubits(*indices: int) -> tuple[tuple[str, int], ...]:
    return tuple(("_ancilla", index) for index in indices)


def _never_reusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No operation waits for any other, so no qubit is ever handed on."""
    monkeypatch.setattr(ancilla_pool._Order, "waits_for", lambda self, qubits: 0)


MULTIPLY_DECLS = "qint<2> a = 1;\nqint<2> b = 3;\nqint<4> c = 0;\n"


def test_a_multiplication_in_a_loop_reuses_one_product_register() -> None:
    """Every iteration writes `c`, so each one already waits for the last."""
    module = _module(MULTIPLY_DECLS + "for(int i in range(4)) {\n    c += a * b;\n}\n")
    assert module.ancilla_size == 4
    assert set(module.ancilla_qubits.values()) == {_pool_qubits(0, 1, 2, 3)}


def test_repeated_padding_reuses_the_same_qubits() -> None:
    module = _module("qint<5> t = 0;\nqint<2> s = 1;\nt += s;\nt += s;\nt += s;\n")
    assert module.ancilla_size == 3
    assert module.ancilla_qubits == {
        "_padding_0": _pool_qubits(0, 1, 2),
        "_padding_1": _pool_qubits(0, 1, 2),
        "_padding_2": _pool_qubits(0, 1, 2),
    }


def test_negated_qifs_in_a_row_share_one_ancilla() -> None:
    module = _module(
        "qint<3> a = [];\nqbool o = false;\n"
        "qif(a != 2) { X(o); }\nqif(a != 3) { X(o); }\nqif(a != 5) { X(o); }\n"
    )
    assert module.ancilla_size == 1


def test_a_clause_ancilla_and_the_whole_negation_never_share() -> None:
    """Both live through the same body, so they are in use at once."""
    module = _module(
        "qint<2> a = [];\nqint<2> b = [];\nqbool o = false;\nqif(!(a != 2 && b == 1)) { X(o); }\n"
    )
    assert module.ancilla_size == 2


def test_independent_multiplications_do_not_share_a_product() -> None:
    """Sharing would make the second wait for the first, which nothing else
    in the program asks for."""
    module = _module(
        "qint<2> a = 1;\nqint<2> b = 3;\nqint<4> c = 0;\n"
        "qint<2> d = 2;\nqint<2> e = 2;\nqint<4> f = 0;\n"
        "c += a * b;\nf += d * e;\n"
    )
    assert module.ancilla_size == 8


def test_reuse_saves_qubits_without_adding_a_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Depth counted in whole operations, the level the pool reasons at; split
    into gates, an operation could have started early on a fresh qubit."""
    source = (
        "qint<2> a = 1;\nqint<2> b = 3;\nqint<4> c = 0;\n"
        "qint<2> d = 2;\nqint<2> e = 2;\nqint<4> f = 0;\n"
        "c += a * b;\nf += d * e;\nc += a * b;\nf += d * e;\n"
    )
    pooled = _circuit(source)
    _never_reusing(monkeypatch)
    unpooled = _circuit(source)
    assert pooled.num_qubits < unpooled.num_qubits
    assert pooled.depth() == unpooled.depth()


def test_a_chain_through_other_registers_counts_as_an_order() -> None:
    """The second multiplication shares no register with the first, but the
    CX ties them together: it waits for the uncomputation, which touches `a`,
    and the second multiplication waits for it through `f`."""
    decls = (
        "qint<2> a = 1;\nqint<2> b = 1;\nqint<4> c = 0;\n"
        "qint<2> f = 0;\nqint<2> h = 1;\nqint<4> g = 0;\n"
    )
    chained = _module(decls + "c += a * b;\nCX(a[1], f[1]);\ng += f * h;\n")
    assert chained.ancilla_size == 4

    unchained = _module(decls + "c += a * b;\nX(f[1]);\ng += f * h;\n")
    assert unchained.ancilla_size == 8


def test_a_product_is_free_only_once_it_is_uncomputed() -> None:
    """The second multiplication waits for the first one's addition, through
    `c`, but not for its uncomputation, which touches only `a`, `b` and the
    product register itself."""
    module = _module(
        "qint<2> a = 1;\nqint<2> b = 1;\nqint<2> c = 0;\nqint<2> e = 1;\nqint<2> d = 0;\n"
        "c += a * b;\nd += c * e;\n"
    )
    assert module.ancilla_size == 8


# Two products keep a qif body wholly controlled, so their registers live in
# the body itself; with one, the product is lifted out in front of the qif.
TWO_PRODUCTS = "{ f += d * e; f += d * e; }"


def test_a_qif_body_inherits_only_what_comes_before_the_qif() -> None:
    module = _module(
        MULTIPLY_DECLS + "qint<2> d = 2;\nqint<2> e = 1;\nqint<4> f = 0;\nqbool k = [];\n"
        f"c += a * b;\nqif(k) {TWO_PRODUCTS}\n"
    )
    assert module.ancilla_size == 8


def test_what_a_qif_body_used_is_free_after_the_qif() -> None:
    module = _module(
        MULTIPLY_DECLS + "qint<2> d = 2;\nqint<2> e = 1;\nqint<4> f = 0;\nqbool k = [];\n"
        f"qif(k) {TWO_PRODUCTS}\nf += d * e;\n"
    )
    assert len(set(module.ancilla_qubits.values())) == 1


def test_an_order_on_one_qubit_says_nothing_about_its_neighbour() -> None:
    """`CX(a[0], f[0])` waits for the uncomputation, but `qif(f[1])` touches
    neither of its qubits, so the qif does not wait for it."""
    module = _module(
        MULTIPLY_DECLS + "qint<2> f = 0;\nqint<2> d = 1;\nqint<2> e = 1;\nqint<4> g = 0;\n"
        "c += a * b;\nCX(a[0], f[0]);\nqif(f[1]) { g += d * e; g += d * e; }\n"
    )
    assert module.ancilla_size == 8


def test_a_qif_body_reuses_its_own_product_register() -> None:
    module = _module(MULTIPLY_DECLS + "qbool k = [];\nqif(k) { c += a * b; c += a * b; }\n")
    assert module.ancilla_size == 4


def test_two_arms_of_a_branch_take_the_same_qubits() -> None:
    """Only one arm ever runs."""
    module = _module(
        MULTIPLY_DECLS + "qint<4> d = 0;\nqbool k = [];\nrt int<> m = measure(k);\n"
        "rt if (m == 1) { c += a * b; } else { d += a * b; }\n"
    )
    assert module.ancilla_size == 4


def test_a_loop_body_reuses_a_qubit_freed_before_the_loop() -> None:
    module = _module(
        MULTIPLY_DECLS + "qbool k = [];\nc += a * b;\nrt int<> m = measure(k);\n"
        "rt while (m == 1) {\n    c += a * b;\n    m = measure(k);\n}\n"
    )
    assert module.ancilla_size == 4


def test_the_qubit_warning_counts_the_pool_not_every_request() -> None:
    """Twelve multiplications ask for 48 scratch qubits; one product's worth
    is ever in use."""
    result = compile_source(
        "qint<4> a = 1;\nqint<4> b = 3;\nqint<8> c = 0;\n"
        "for(int i in range(12)) {\n    c += a * b;\n}\n",
        source_name="test.slanq",
    )
    assert not result.diagnostics.warnings
    assert result.ir is not None
    assert result.ir.ancilla_size == 8


def test_a_register_used_both_outside_and_inside_a_block_lives_through_both(
    span: Span,
) -> None:
    """Built by hand: no lowering emits this today, since every scratch
    register is born and dies in one place.

    `_product_0` is used by the addition and again inside the qif, so it lives
    through both. Counted as living only through the addition, its qubits would
    be free by the qif, and the body would hand it fresh ones -- those
    `_padding_0` held during the addition, putting the two on the same qubits."""
    c = QubitRef(name="c", size=4)
    d = QubitRef(name="d", size=2)
    k = QubitRef(name="k", size=1)
    padding = QubitRef(name="_padding_0", size=2)
    temp = QubitRef(name="_product_0", size=2)
    module = IRModule(
        qubits=[c, d, k],
        body=IRBlock(
            ops=[
                DeclareAncillaOp(span=span, ref=padding),
                DeclareAncillaOp(span=span, ref=temp),
                ArithmeticOp(span=span, target=c, addend=[padding, temp], subtract=False),
                QIfOp(
                    span=span,
                    direct_qubits=[QubitBit(ref=k, index=0)],
                    ctrl_state=1,
                    clause_ancillas=[],
                    negated=False,
                    ancilla=None,
                    body=IRBlock(
                        ops=[ArithmeticOp(span=span, target=d, addend=[temp], subtract=False)]
                    ),
                ),
            ]
        ),
    )
    ancilla_pool.allocate_ancillas(module)
    assert not set(module.ancilla_qubits["_padding_0"]) & set(module.ancilla_qubits["_product_0"])


def test_both_arms_of_a_branch_share_a_register_they_both_use(span: Span) -> None:
    """Built by hand, like the test above. Placed in the first arm and then
    skipped in the second, `_product_0` would be free in the second arm, which
    would hand its qubits to `_padding_0` beside it."""
    c = QubitRef(name="c", size=2)
    d = QubitRef(name="d", size=4)
    padding = QubitRef(name="_padding_0", size=2)
    temp = QubitRef(name="_product_0", size=2)
    module = IRModule(
        qubits=[c, d],
        body=IRBlock(
            ops=[
                DeclareAncillaOp(span=span, ref=padding),
                DeclareAncillaOp(span=span, ref=temp),
                RealtimeIfOp(
                    span=span,
                    condition=Literal(span=span, value=True),
                    body=IRBlock(
                        ops=[ArithmeticOp(span=span, target=c, addend=[temp], subtract=False)]
                    ),
                    orelse=IRBlock(
                        ops=[
                            ArithmeticOp(
                                span=span, target=d, addend=[padding, temp], subtract=False
                            )
                        ]
                    ),
                ),
            ]
        ),
    )
    ancilla_pool.allocate_ancillas(module)
    assert not set(module.ancilla_qubits["_padding_0"]) & set(module.ancilla_qubits["_product_0"])


TWO_QIF_PRODUCTS = (
    "qint<2> a = [];\nqint<2> b = [];\nqint<2> d = [];\nqint<2> e = [];\n"
    "qint<4> c = 0;\nqint<4> f = 0;\nqbool k = [];\n"
    "qif(k) { c += a * b; }\nqif(k) { f += d * e; }\n"
)


def test_a_lifted_multiplication_waits_only_for_what_it_touches() -> None:
    """The second qif's multiplication runs in front of it, uncontrolled, and
    touches nothing the first product's uncomputation touches, so it does not
    wait for it -- and does not get its qubits."""
    assert _module(TWO_QIF_PRODUCTS).ancilla_size == 8


def test_a_negation_ancilla_waits_only_for_its_own_clause() -> None:
    """The clause `x != 2` is computed from `x` alone. The qif's body touches
    `y`, which the product's uncomputation also touches, but the clause
    ancilla's first instruction does not."""
    assert _module(CLAUSE_AFTER_PRODUCT).ancilla_size == 5


CLAUSE_AFTER_PRODUCT = (
    "qint<2> w = 0;\nqint<2> y = [];\nqint<2> z = 1;\nqint<2> x = [];\nqbool o = false;\n"
    "w += y * z;\nqif(x != 2) { CX(y[0], o); }\n"
)


# The whole condition is negated into one ancilla, which is first computed
# from the condition's qubits -- not from everything the qif touches.
WHOLE_NEGATION_AFTER_PRODUCT = CLAUSE_AFTER_PRODUCT.replace("x != 2", "!(x == 2)")


def test_a_whole_negation_ancilla_waits_only_for_the_condition() -> None:
    assert _module(WHOLE_NEGATION_AFTER_PRODUCT).ancilla_size == 5


@pytest.mark.parametrize(
    "source", [TWO_QIF_PRODUCTS, CLAUSE_AFTER_PRODUCT, WHOLE_NEGATION_AFTER_PRODUCT]
)
def test_a_qif_adds_no_wait_between_operations(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A qif becomes several instructions; none of them may wait for more than
    it did with every register on fresh qubits."""
    pooled = _circuit(source)
    _never_reusing(monkeypatch)
    unpooled = _circuit(source)
    # These programs save no qubits; what they pin down is the depth.
    assert pooled.num_qubits == unpooled.num_qubits
    assert pooled.depth() == unpooled.depth()


# `b`'s history is two layers deep -- the initialiser, then the H -- so the
# multiplication has to be three layers out before it may take b's qubits. The
# X gates cancel each other and are there only to push it that far.
DEAD_THEN_CHAIN = (
    "qint<4> b = 5;\nH(b[0]);\nqint<2> a = 1;\nqint<2> d = 3;\nqint<4> c = 0;\n"
    "X(a[0]);\nX(a[0]);\nX(a[0]);\nc += a * d;\n"
)


def _uncomputed(module: IRModule) -> list[str]:
    return [op.target.name for op in module.body.ops if isinstance(op, UncomputeOp)]


def test_a_dead_variable_lends_its_qubits_once_it_is_undone() -> None:
    """`b` is never entangled with anything, and the multiplication could not
    start before its own layer anyway, by which `b`'s last use and the
    uncomputation after it are over."""
    module = _module(DEAD_THEN_CHAIN)
    assert module.ancilla_size == 0
    assert module.ancilla_qubits["_product_0"] == (("b", 0), ("b", 1), ("b", 2), ("b", 3))
    assert _uncomputed(module) == ["b"]
    ops = module.body.ops
    uncompute = next(op for op in ops if isinstance(op, UncomputeOp))
    assert ops[ops.index(uncompute) - 1] is uncompute.history[-1]


def test_a_dead_variable_does_not_hold_back_an_early_operation() -> None:
    """The multiplication could run in the second layer; `b` is last used in
    the third, so taking its qubits would make the multiplication wait."""
    module = _module(
        "qint<4> b = 5;\nH(b[0]);\nX(b[1]);\nqint<2> a = 1;\nqint<2> d = 3;\nqint<4> c = 0;\n"
        "c += a * d;\n"
    )
    assert module.ancilla_size == 4
    assert _uncomputed(module) == []


@pytest.mark.parametrize("gates", ["X(a[0]);\n", "X(a[0]);\nX(a[0]);\n"])
def test_the_uncomputation_costs_as_many_layers_as_it_takes(gates: str) -> None:
    """The uncomputation is charged its own depth, not a flat one layer. `b`'s
    history is two layers, so the multiplication two layers out is still too
    early -- it is the case that would lend if the cost were a flat one."""
    module = _module(DEAD_THEN_CHAIN.replace("X(a[0]);\n" * 3, gates))
    assert module.ancilla_size == 4
    assert _uncomputed(module) == []


def test_a_one_layer_history_costs_one_layer() -> None:
    """Without the initialiser `b`'s history is the H alone, so two layers of
    room are enough -- the same program that is refused above."""
    module = _module(
        DEAD_THEN_CHAIN.replace("qint<4> b = 5;", "qint<4> b = 0;").replace(
            "X(a[0]);\n" * 3, "X(a[0]);\n" * 2
        )
    )
    assert module.ancilla_size == 0
    assert _uncomputed(module) == ["b"]


def test_the_compilers_own_qubits_go_before_a_dead_variables() -> None:
    module = _module(DEAD_THEN_CHAIN + "c += a * d;\n")
    assert module.ancilla_qubits["_product_1"] == module.ancilla_qubits["_product_0"]
    assert _uncomputed(module) == ["b"]


def test_a_variable_ending_in_a_reset_lends_its_qubits_without_uncomputation() -> None:
    module = _module(DEAD_THEN_CHAIN.replace("H(b[0]);", "reset(b);"))
    assert module.ancilla_qubits["_product_0"] == (("b", 0), ("b", 1), ("b", 2), ("b", 3))
    assert _uncomputed(module) == []


def test_a_variable_nothing_touches_lends_its_qubits_from_the_start() -> None:
    module = _module(
        "qint<4> b = 0;\nqint<2> a = 1;\nqint<2> d = 3;\nqint<4> c = 0;\nc += a * d;\n"
    )
    assert module.ancilla_size == 0
    assert _uncomputed(module) == []


def test_two_registers_alive_at_once_never_share_a_qubit() -> None:
    """A lent variable wide enough for both products must still hand each of
    them its own qubits."""
    module = _module(
        "qint<8> b = 5;\nH(b[0]);\nqint<2> a = 1;\nqint<2> d = 3;\n"
        "qint<4> c = 0;\nqint<4> g = 0;\n" + "X(a[0]);\n" * 3 + "c += a * d;\ng += a * d;\n"
    )
    first = set(module.ancilla_qubits["_product_0"])
    second = set(module.ancilla_qubits["_product_1"])
    assert len(first) == len(second) == 4
    assert not first & second


def test_a_variable_built_from_a_rewritten_classical_is_not_lent() -> None:
    """The uncomputation replays the history by rendering its expressions
    again, which would read the variable's later value."""
    rewritten = _module(
        "int n = 1;\nqbool b = false;\nRY(n * PI / 4, b);\nn = 3;\nRY(n * PI / 4, b);\n"
        "qint<3> t = 0;\nqint<1> s = [];\n" + "X(t[0]);\n" * 4 + "t += s;\n"
    )
    assert _uncomputed(rewritten) == []

    once = _module(
        "int k = 2;\nqbool b = false;\nRY(k * PI / 4, b);\n"
        "qint<3> t = 0;\nqint<1> s = [];\n" + "X(t[0]);\n" * 4 + "t += s;\n"
    )
    assert _uncomputed(once) == ["b"]
