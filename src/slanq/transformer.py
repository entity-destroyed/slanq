from __future__ import annotations

from lark import Transformer, v_args
from lark.tree import Meta

from slanq.ast_nodes import (
    Call,
    ClassicalDecl,
    Expression,
    ExprStatement,
    IntType,
    Literal,
    Name,
    Program,
    QBoolType,
    QuantumDecl,
    Span,
    Statement,
    Type,
)

CTYPE_MAP: dict[str, Type] = {
    "int": IntType(),
}


def _span_from_meta(meta: Meta) -> Span:
    return Span(
        start_line=meta.line,
        start_col=meta.column,
        end_line=meta.end_line,
        end_col=meta.end_column,
    )


def _span_from_token(token) -> Span:
    return Span(
        start_line=token.line,
        start_col=token.column,
        end_line=token.end_line,
        end_col=token.end_column,
    )


class SlanqTransformer(Transformer):
    @v_args(meta=True)
    def start(self, meta: Meta, children: list[Statement]) -> Program:
        return Program(span=_span_from_meta(meta), statements=list(children))

    @v_args(meta=True)
    def expr_statement(self, meta: Meta, children) -> ExprStatement:
        (expr,) = children
        return ExprStatement(span=_span_from_meta(meta), expr=expr)

    @v_args(meta=True)
    def qbool_decl(self, meta: Meta, children) -> QuantumDecl:
        name_token, initializer = children
        return QuantumDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=QBoolType(),
            initializer=initializer,
        )

    @v_args(meta=True)
    def classical_decl(self, meta: Meta, children) -> ClassicalDecl:
        ctype_token, name_token, initializer = children
        return ClassicalDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=CTYPE_MAP[str(ctype_token)],
            initializer=initializer,
        )

    @v_args(meta=True)
    def call_expr(self, meta: Meta, children) -> Call:
        # `[arg_list]` always yields a slot, holding None for a no-argument call.
        name_token, arg_list = children
        return Call(
            span=_span_from_meta(meta),
            callee=Name(span=_span_from_token(name_token), name=str(name_token)),
            args=arg_list or [],
        )

    def arg_list(self, children) -> list[Expression]:
        return list(children)

    @v_args(meta=True)
    def var(self, meta: Meta, children) -> Name:
        (name_token,) = children
        return Name(span=_span_from_meta(meta), name=str(name_token))

    @v_args(meta=True)
    def number(self, meta: Meta, children) -> Literal:
        (num_token,) = children
        text = str(num_token)
        value: int | float = float(text) if "." in text else int(text)
        return Literal(span=_span_from_meta(meta), value=value)

    @v_args(meta=True)
    def boolean(self, meta: Meta, children) -> Literal:
        (bool_token,) = children
        return Literal(span=_span_from_meta(meta), value=(str(bool_token) == "true"))


def transform_to_ast(parse_tree) -> Program:
    return SlanqTransformer().transform(parse_tree)


__all__ = ["SlanqTransformer", "transform_to_ast"]
