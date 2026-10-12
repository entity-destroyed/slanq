from __future__ import annotations

import pytest

from slanq.compiler import compile_source
from slanq.entanglement import EntanglementGraph, Reuse
from slanq.ir import ArithmeticOp, GateOp, InitOp, ResetOp


def _graph(source: str) -> EntanglementGraph:
    result = compile_source(source, source_name="test.slanq", stop_after="ir")
    assert not result.diagnostics.has_errors, [str(d) for d in result.diagnostics]
    assert result.ir is not None
    return EntanglementGraph.build(result.ir.body.ops, result.ir.qubits)


def test_a_node_keeps_its_variable_and_what_touched_it_in_order() -> None:
    graph = _graph("qint<2> a = 1;\nqint<2> b = 0;\nH(a[0]);\nCX(a[1], b[0]);\n")
    # a[0] is the most significant qubit, index 1 in Qiskit's order.
    node = graph.nodes[("a", 1)]
    assert (node.register, node.index) == ("a", 1)
    assert [(touch.seq, type(touch.op), touch.role) for touch in node.history] == [
        (0, InitOp, "write"),
        (2, GateOp, "write"),
    ]


def test_a_declaration_as_zero_touches_nothing() -> None:
    assert _graph("qint<2> b = 0;\n").nodes[("b", 0)].history == []


def test_an_addend_is_read_and_the_target_written() -> None:
    graph = _graph("qint<2> a = 1;\nqint<2> b = 2;\na += b;\n")
    (touch,) = [touch for touch in graph.nodes[("b", 1)].history if touch.seq == 2]
    assert (type(touch.op), touch.role) == (ArithmeticOp, "read")
    (touch,) = [touch for touch in graph.nodes[("a", 0)].history if touch.seq == 2]
    assert touch.role == "write"


def test_an_operation_on_two_variables_joins_them() -> None:
    graph = _graph("qint<2> a = 1;\nqint<2> b = 0;\nCX(a[1], b[0]);\n")
    assert graph.edges[("a", 0)] == {("b", 1)}
    assert graph.variable_edges() == {frozenset({"a", "b"})}


@pytest.mark.parametrize("ending", ["rt int<> m = measure(x);", "reset(x);"])
def test_measuring_or_resetting_a_whole_variable_cuts_its_edges(ending: str) -> None:
    graph = _graph(f"qbool x = [];\nqbool y = false;\nCX(x, y);\n{ending}\n")
    assert graph.edges[("x", 0)] == set()
    assert graph.edges[("y", 0)] == set()
    assert graph.variable_edges() == set()


def test_a_variable_nothing_touches_is_free_from_the_start() -> None:
    assert _graph("qint<2> b = 0;\n").reusable("b") == Reuse(last_use=None, history=())


def test_an_isolated_variable_is_undone_by_its_own_history() -> None:
    reuse = _graph("qint<2> b = 1;\nH(b[0]);\nqif(b[0]) { X(b[1]); }\n").reusable("b")
    assert reuse is not None
    assert reuse.last_use == 2
    assert [type(op).__name__ for op in reuse.history] == ["InitOp", "GateOp", "QIfOp"]
    assert reuse.needs_uncompute


def test_a_reset_starts_the_history_over() -> None:
    graph = _graph("qbool o = false;\nqint<2> b = [];\nCX(b[0], o);\nreset(b);\nX(b[1]);\n")
    reuse = graph.reusable("b")
    assert reuse is not None
    assert [type(op).__name__ for op in reuse.history] == ["GateOp"]


def test_resetting_a_one_qubit_variable_resets_all_of_it() -> None:
    """A qbool reaches the reset as its single bit, which is still the whole
    variable."""
    graph = _graph("qbool b = [];\nqbool o = false;\nCX(b, o);\nreset(b);\nX(b);\n")
    reuse = graph.reusable("b")
    assert reuse is not None
    assert [type(op).__name__ for op in reuse.history] == ["GateOp"]
    assert graph.variable_edges() == set()


def test_a_variable_ending_in_a_reset_needs_no_uncomputation() -> None:
    graph = _graph("qint<2> b = 3;\nreset(b);\n")
    reuse = graph.reusable("b")
    assert reuse == Reuse(last_use=1, history=())
    assert isinstance(graph.nodes[("b", 0)].history[-1].op, ResetOp)


@pytest.mark.parametrize(
    "source",
    [
        "qint<2> b = 1;\nqint<2> a = 0;\na += b;\n",
        "qbool b = [];\nqbool o = false;\nCX(b, o);\n",
        "qint<2> b = 1;\nrt int<> m = measure(b);\n",
        "qbool f = [];\nrt int<> m = measure(f);\nqint<2> b = 0;\nrt if (m == 1) { X(b[0]); }\n",
        "qint<2> b = 1;\nreset(b[0]);\n",
    ],
    ids=["read by another", "gate with another", "measured", "written at run time", "half reset"],
)
def test_a_variable_that_cannot_be_undone_unseen_is_not_reusable(source: str) -> None:
    assert _graph(source).reusable("b") is None


def test_a_moved_variable_is_not_reusable() -> None:
    graph = _graph("qint<2> n = 1;\nqubit[2] q = to_qubits(n);\nH(q[0]);\n")
    assert graph.reusable("n") is None
