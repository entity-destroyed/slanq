from __future__ import annotations

from slanq.ast_nodes import (
    Call,
    ClassicalDecl,
    Declaration,
    ExprStatement,
    IntType,
    Literal,
    Name,
    Program,
    QBoolType,
    QuantumDecl,
    Span,
)


def test_span_carries_start_and_end() -> None:
    s = Span(start_line=2, start_col=3, end_line=4, end_col=5)
    assert (s.start_line, s.start_col, s.end_line, s.end_col) == (2, 3, 4, 5)


def test_types_compare_by_value() -> None:
    assert QBoolType() == QBoolType()
    assert IntType() == IntType()
    assert QBoolType() != IntType()


def test_name_defaults_have_no_annotations(span: Span) -> None:
    node = Name(span=span, name="q")
    assert node.inferred_type is None
    assert node.resolved_symbol is None


def test_name_is_mutable_for_semantic_annotations(span: Span) -> None:
    node = Name(span=span, name="q")
    node.inferred_type = QBoolType()
    assert node.inferred_type == QBoolType()


def test_literal_carries_python_value(span: Span) -> None:
    assert Literal(span=span, value=False).value is False


def test_program_default_statements_lists_are_independent(span: Span) -> None:
    a = Program(span=span)
    b = Program(span=span)
    a.statements.append(ExprStatement(span=span, expr=Name(span=span, name="x")))
    assert len(a.statements) == 1
    assert len(b.statements) == 0


def test_call_with_multiple_args(span: Span) -> None:
    call = Call(
        span=span,
        callee=Name(span=span, name="CX"),
        args=[Name(span=span, name="a"), Name(span=span, name="b")],
    )
    assert call.callee.name == "CX"
    assert len(call.args) == 2


def test_decls_share_the_declaration_base(span: Span) -> None:
    quantum = QuantumDecl(
        span=span,
        name="q",
        declared_type=QBoolType(),
        initializer=Literal(span=span, value=False),
    )
    classical = ClassicalDecl(
        span=span,
        name="r",
        declared_type=IntType(),
        initializer=Literal(span=span, value=0),
    )
    assert isinstance(quantum, Declaration)
    assert isinstance(classical, Declaration)


def test_mvp_a_is_representable(mvp_a_ast: Program) -> None:
    assert len(mvp_a_ast.statements) == 3
    declaration = mvp_a_ast.statements[0]
    assert isinstance(declaration, QuantumDecl)
    assert declaration.name == "q"
