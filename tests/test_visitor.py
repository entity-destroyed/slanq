from __future__ import annotations

from slanq.ast_nodes import (
    Call,
    ClassicalDecl,
    ExprStatement,
    IntType,
    Literal,
    Name,
    Program,
    QBoolType,
    QuantumDecl,
    Span,
)
from slanq.visitor import NodeVisitor, iter_child_nodes


def _s() -> Span:
    return Span(start_line=1, start_col=1, end_line=1, end_col=1)


def _mvp_a_ast() -> Program:
    s = _s()
    return Program(
        span=s,
        statements=[
            QuantumDecl(
                span=s,
                name="q",
                declared_type=QBoolType(),
                initializer=Literal(span=s, value=False),
            ),
            ExprStatement(
                span=s,
                expr=Call(
                    span=s,
                    callee=Name(span=s, name="H"),
                    args=[Name(span=s, name="q")],
                ),
            ),
            ClassicalDecl(
                span=s,
                name="result",
                declared_type=IntType(),
                initializer=Call(
                    span=s,
                    callee=Name(span=s, name="measure"),
                    args=[Name(span=s, name="q")],
                ),
            ),
        ],
    )


def test_iter_child_nodes_yields_direct_children() -> None:
    prog = _mvp_a_ast()
    children = list(iter_child_nodes(prog))
    assert len(children) == 3


def test_iter_child_nodes_skips_type_and_span() -> None:
    (decl, _, _) = _mvp_a_ast().statements
    children = list(iter_child_nodes(decl))
    assert children == [decl.initializer]


def test_generic_visit_walks_entire_tree() -> None:
    visited: list[str] = []

    class Recorder(NodeVisitor):
        def generic_visit(self, node):
            visited.append(type(node).__name__)
            super().generic_visit(node)

    Recorder().visit(_mvp_a_ast())
    assert visited[0] == "Program"
    assert visited.count("Name") == 4
    assert visited.count("Literal") == 1


def test_specific_visit_method_wins_over_generic() -> None:
    seen: list[str] = []

    class NameCollector(NodeVisitor):
        def visit_Name(self, node):
            seen.append(node.name)
            self.generic_visit(node)

    NameCollector().visit(_mvp_a_ast())
    assert set(seen) == {"H", "q", "measure"}
    assert seen.count("q") == 2


def test_visit_returns_value_from_specific_method() -> None:
    class ConstReturner(NodeVisitor):
        def visit_Literal(self, node):
            return "literal-value"

    result = ConstReturner().visit(Literal(span=_s(), value=42))
    assert result == "literal-value"
