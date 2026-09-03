from __future__ import annotations

from lark import Token, Transformer, v_args
from lark.tree import Meta

from slanq.ast_nodes import (
    AmplitudeList,
    Assign,
    AugAssign,
    BinaryOp,
    Block,
    BoolType,
    Call,
    ClassicalDecl,
    ComplexType,
    Expression,
    ExprStatement,
    FloatType,
    For,
    If,
    Index,
    IntType,
    Literal,
    LoopVarDecl,
    Name,
    ParamArrayDecl,
    ParamDecl,
    ProbList,
    ProcessDef,
    ProcParam,
    Program,
    QBoolType,
    QIf,
    QIntType,
    QuantumDecl,
    Span,
    Statement,
    Type,
    UnaryOp,
    While,
)
from slanq.diagnostics import SlanqError

TYPE_BY_NAME: dict[str, type[Type]] = {
    "int": IntType,
    "float": FloatType,
    "bool": BoolType,
    "complex": ComplexType,
    "qint": QIntType,
    "qbool": QBoolType,
}


def _span_from_meta(meta: Meta) -> Span:
    return Span(
        start_line=meta.line,
        start_col=meta.column,
        end_line=meta.end_line,
        end_col=meta.end_column,
    )


def _span_from_token(token: Token) -> Span:
    return Span(
        start_line=token.line,
        start_col=token.column,
        end_line=token.end_line,
        end_col=token.end_column,
    )


def _name_from_token(token: Token) -> Name:
    return Name(span=_span_from_token(token), name=str(token))


def _type_from_token(token: Token) -> Type:
    return TYPE_BY_NAME[str(token)]()


@v_args(meta=True)
class SlanqTransformer(Transformer):
    def __default__(self, data, children, meta):
        """Lark would otherwise rebuild the node as a plain Tree, which every
        later phase silently skips. Failing loudly keeps grammar and transformer
        from drifting apart."""
        where = "" if meta.empty else f" at line {meta.line}"
        raise SlanqError(
            f"internal error: grammar rule '{data}' has no AST transformation{where}. "
            "This is a bug in the compiler, not in the source program."
        )

    def start(self, meta: Meta, children: list[Statement]) -> Program:
        return Program(span=_span_from_meta(meta), statements=list(children))

    def block(self, meta: Meta, children: list[Statement]) -> Block:
        return Block(span=_span_from_meta(meta), statements=list(children))

    # --- declarations ---

    def qint_decl(self, meta: Meta, children) -> QuantumDecl:
        width_token, name_token, initializer = children
        return QuantumDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=QIntType(size=int(width_token)),
            initializer=initializer,
        )

    def qbool_decl(self, meta: Meta, children) -> QuantumDecl:
        name_token, initializer = children
        return QuantumDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=QBoolType(),
            initializer=initializer,
        )

    def classical_decl(self, meta: Meta, children) -> ClassicalDecl:
        type_token, name_token, initializer = children
        return ClassicalDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=_type_from_token(type_token),
            initializer=initializer,
        )

    def param_decl(self, meta: Meta, children) -> ParamDecl:
        type_token, name_token = children
        return ParamDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=_type_from_token(type_token),
        )

    def param_array_decl(self, meta: Meta, children) -> ParamArrayDecl:
        type_token, name_token, size_token = children
        return ParamArrayDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=_type_from_token(type_token),
            size=int(size_token),
        )

    def prob_list(self, meta: Meta, children) -> ProbList:
        # An empty list yields a single None placeholder rather than no children.
        return ProbList(
            span=_span_from_meta(meta),
            probabilities=[float(token) for token in children if token is not None],
        )

    def amplitude_list(self, meta: Meta, children) -> AmplitudeList:
        # An empty list yields a single None placeholder rather than no children.
        return AmplitudeList(
            span=_span_from_meta(meta),
            elements=[element for element in children if element is not None],
        )

    # --- procedures ---

    def process_def(self, meta: Meta, children) -> ProcessDef:
        name_token, params, body = children
        return ProcessDef(
            span=_span_from_meta(meta),
            name=str(name_token),
            params=params or [],
            body=body,
        )

    def param_list(self, meta: Meta, children) -> list[ProcParam]:
        return list(children)

    def proc_param(self, meta: Meta, children) -> ProcParam:
        type_token, name_token = children
        return ProcParam(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=_type_from_token(type_token),
        )

    # --- statements ---

    def expr_statement(self, meta: Meta, children) -> ExprStatement:
        (expr,) = children
        return ExprStatement(span=_span_from_meta(meta), expr=expr)

    def assignment(self, meta: Meta, children) -> Assign:
        target, value = children
        return Assign(span=_span_from_meta(meta), target=target, value=value)

    def compound_assignment(self, meta: Meta, children) -> AugAssign:
        target, op_token, value = children
        return AugAssign(
            span=_span_from_meta(meta),
            target=target,
            op=str(op_token),
            value=value,
        )

    def if_stmt(self, meta: Meta, children) -> If:
        condition, body = children
        return If(span=_span_from_meta(meta), condition=condition, body=body)

    def qif_stmt(self, meta: Meta, children) -> QIf:
        condition, body = children
        return QIf(span=_span_from_meta(meta), condition=condition, body=body)

    def while_stmt(self, meta: Meta, children) -> While:
        condition, body = children
        return While(span=_span_from_meta(meta), condition=condition, body=body)

    def for_stmt(self, meta: Meta, children) -> For:
        binding, iterable, body = children
        return For(
            span=_span_from_meta(meta),
            binding=binding,
            iterable=iterable,
            body=body,
        )

    def loop_var(self, meta: Meta, children) -> LoopVarDecl:
        type_token, name_token = children
        return LoopVarDecl(
            span=_span_from_meta(meta),
            name=str(name_token),
            declared_type=_type_from_token(type_token),
        )

    # --- expressions ---

    def call_expr(self, meta: Meta, children) -> Call:
        # `[arg_list]` always yields a slot, holding None for a no-argument call.
        name_token, arg_list = children
        return Call(
            span=_span_from_meta(meta),
            callee=_name_from_token(name_token),
            args=arg_list or [],
        )

    def arg_list(self, meta: Meta, children) -> list[Expression]:
        return list(children)

    def index(self, meta: Meta, children) -> Index:
        name_token, index = children
        return Index(
            span=_span_from_meta(meta),
            base=_name_from_token(name_token),
            index=index,
        )

    def var(self, meta: Meta, children) -> Name:
        (name_token,) = children
        return _name_from_token(name_token)

    def number(self, meta: Meta, children) -> Literal:
        (token,) = children
        text = str(token)
        value: int | float = float(text) if "." in text else int(text)
        return Literal(span=_span_from_meta(meta), value=value)

    def imaginary(self, meta: Meta, children) -> Literal:
        (token,) = children
        magnitude = float(str(token)[:-1])  # strip the trailing "i"
        return Literal(span=_span_from_meta(meta), value=complex(0.0, magnitude))

    def boolean(self, meta: Meta, children) -> Literal:
        (token,) = children
        return Literal(span=_span_from_meta(meta), value=str(token) == "true")

    def _binary(self, meta: Meta, children, op: str) -> BinaryOp:
        left, right = children
        return BinaryOp(span=_span_from_meta(meta), op=op, left=left, right=right)

    def _unary(self, meta: Meta, children, op: str) -> UnaryOp:
        (operand,) = children
        return UnaryOp(span=_span_from_meta(meta), op=op, operand=operand)

    def logical_or(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "||")

    def logical_and(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "&&")

    def bit_or(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "|")

    def bit_xor(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "^")

    def bit_and(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "&")

    def eq(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "==")

    def ne(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "!=")

    def le(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "<=")

    def ge(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, ">=")

    def lt(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "<")

    def gt(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, ">")

    def add(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "+")

    def sub(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "-")

    def mul(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "*")

    def div(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "/")

    def mod(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "%")

    def pow(self, meta: Meta, children) -> BinaryOp:
        return self._binary(meta, children, "**")

    def neg(self, meta: Meta, children) -> UnaryOp:
        return self._unary(meta, children, "-")

    def bit_not(self, meta: Meta, children) -> UnaryOp:
        return self._unary(meta, children, "~")

    def logical_not(self, meta: Meta, children) -> UnaryOp:
        return self._unary(meta, children, "!")


def transform_to_ast(parse_tree) -> Program:
    return SlanqTransformer().transform(parse_tree)


__all__ = ["SlanqTransformer", "transform_to_ast"]
