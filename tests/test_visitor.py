from __future__ import annotations

from slanq.ast_nodes import Literal, Program, QuantumDecl, Span
from slanq.visitor import NodeVisitor, iter_child_nodes


def test_iter_child_nodes_yields_direct_children(mvp_a_ast: Program) -> None:
    assert len(list(iter_child_nodes(mvp_a_ast))) == 3


def test_iter_child_nodes_skips_span_and_type(mvp_a_ast: Program) -> None:
    declaration = mvp_a_ast.statements[0]
    assert isinstance(declaration, QuantumDecl)
    assert list(iter_child_nodes(declaration)) == [declaration.initializer]


def test_generic_visit_walks_entire_tree(mvp_a_ast: Program) -> None:
    visited: list[str] = []

    class Recorder(NodeVisitor):
        def generic_visit(self, node):
            visited.append(type(node).__name__)
            super().generic_visit(node)

    Recorder().visit(mvp_a_ast)
    assert visited[0] == "Program"
    assert visited.count("Name") == 4
    assert visited.count("Literal") == 1


def test_specific_visit_method_wins_over_generic(mvp_a_ast: Program) -> None:
    seen: list[str] = []

    class NameCollector(NodeVisitor):
        def visit_Name(self, node):
            seen.append(node.name)
            self.generic_visit(node)

    NameCollector().visit(mvp_a_ast)
    assert set(seen) == {"H", "q", "measure"}
    assert seen.count("q") == 2


def test_visit_returns_value_from_specific_method(span: Span) -> None:
    class ConstReturner(NodeVisitor):
        def visit_Literal(self, node):
            return "literal-value"

    assert ConstReturner().visit(Literal(span=span, value=42)) == "literal-value"
