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


def _span(line: int = 1, col: int = 1) -> Span:
    return Span(start_line=line, start_col=col, end_line=line, end_col=col + 1)


def test_span_carries_start_and_end() -> None:
    s = Span(start_line=2, start_col=3, end_line=4, end_col=5)
    assert (s.start_line, s.start_col, s.end_line, s.end_col) == (2, 3, 4, 5)


def test_types_compare_by_value() -> None:
    assert QBoolType() == QBoolType()
    assert IntType() == IntType()
    assert QBoolType() != IntType()


def test_name_defaults_have_no_annotations() -> None:
    n = Name(span=_span(), name="q")
    assert n.inferred_type is None
    assert n.resolved_symbol is None


def test_name_is_mutable_for_semantic_annotations() -> None:
    n = Name(span=_span(), name="q")
    n.inferred_type = QBoolType()
    assert n.inferred_type == QBoolType()


def test_literal_carries_python_value() -> None:
    lit = Literal(span=_span(), value=False)
    assert lit.value is False


def test_program_default_statements_lists_are_independent() -> None:
    a = Program(span=_span())
    b = Program(span=_span())
    a.statements.append(ExprStatement(span=_span(), expr=Name(span=_span(), name="x")))
    assert len(a.statements) == 1
    assert len(b.statements) == 0


def test_call_with_multiple_args() -> None:
    call = Call(
        span=_span(),
        callee=Name(span=_span(), name="CX"),
        args=[
            Name(span=_span(), name="a"),
            Name(span=_span(), name="b"),
        ],
    )
    assert call.callee.name == "CX"
    assert len(call.args) == 2


def test_decls_are_declarations() -> None:
    s = _span()
    qd = QuantumDecl(
        span=s,
        name="q",
        declared_type=QBoolType(),
        initializer=Literal(span=s, value=False),
    )
    cd = ClassicalDecl(
        span=s,
        name="r",
        declared_type=IntType(),
        initializer=Literal(span=s, value=0),
    )
    assert isinstance(qd, Declaration)
    assert isinstance(cd, Declaration)


def test_mvp_a_ast_can_be_hand_built() -> None:
    s = _span()
    program = Program(
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
    assert len(program.statements) == 3
    decl = program.statements[0]
    assert isinstance(decl, QuantumDecl)
    assert decl.name == "q"
