from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slanq.builtin import Signature


@dataclass(kw_only=True)
class Span:
    """Half-open source range: [start_line:start_col, end_line:end_col)."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass(kw_only=True)
class Type:
    pass


@dataclass(kw_only=True)
class QBoolType(Type):
    pass


@dataclass(kw_only=True)
class QIntType(Type):
    # Unsized in a process parameter list, where `qint` carries no width.
    size: int | None = None


def is_quantum(declared: Type) -> bool:
    return isinstance(declared, QBoolType | QIntType)


def qubit_count(declared: Type) -> int | None:
    if isinstance(declared, QBoolType):
        return 1
    if isinstance(declared, QIntType):
        return declared.size
    return None


@dataclass(kw_only=True)
class IntType(Type):
    pass


@dataclass(kw_only=True)
class FloatType(Type):
    pass


@dataclass(kw_only=True)
class BoolType(Type):
    pass


@dataclass(kw_only=True)
class ComplexType(Type):
    pass


@dataclass(kw_only=True)
class Node:
    span: Span


@dataclass(kw_only=True)
class Statement(Node):
    pass


@dataclass(kw_only=True)
class Expression(Node):
    inferred_type: Type | None = None


@dataclass(kw_only=True)
class Declaration(Statement):
    name: str


@dataclass(kw_only=True)
class Program(Node):
    statements: list[Statement] = field(default_factory=list)


@dataclass(kw_only=True)
class Block(Node):
    statements: list[Statement] = field(default_factory=list)


@dataclass(kw_only=True)
class ProbList(Node):
    probabilities: list[float] = field(default_factory=list)


@dataclass(kw_only=True)
class AmplitudeList(Node):
    """`{}`-syntax amplitude list. Unlike ProbList, elements are arbitrary
    compile-time expressions (not pre-evaluated), since they may be complex
    and are only resolved to values later, via const_value."""

    elements: list[Expression] = field(default_factory=list)


@dataclass(kw_only=True)
class QuantumDecl(Declaration):
    declared_type: Type
    initializer: Expression | ProbList | AmplitudeList


@dataclass(kw_only=True)
class ClassicalDecl(Declaration):
    declared_type: Type
    initializer: Expression


@dataclass(kw_only=True)
class ParamDecl(Declaration):
    declared_type: Type


@dataclass(kw_only=True)
class ParamArrayDecl(Declaration):
    declared_type: Type
    size: int


@dataclass(kw_only=True)
class BuiltinDecl(Declaration):
    """A gate, function or constant the language provides itself. Not built from
    source, so it carries a placeholder span."""

    declared_type: Type | None = None
    signature: Signature | None = None


@dataclass(kw_only=True)
class ProcParam(Node):
    name: str
    declared_type: Type


@dataclass(kw_only=True)
class ProcessDef(Declaration):
    params: list[ProcParam] = field(default_factory=list)
    body: Block


@dataclass(kw_only=True)
class ExprStatement(Statement):
    expr: Expression


@dataclass(kw_only=True)
class Assign(Statement):
    target: Expression
    value: Expression


@dataclass(kw_only=True)
class AugAssign(Statement):
    target: Expression
    op: str
    value: Expression


@dataclass(kw_only=True)
class If(Statement):
    condition: Expression
    body: Block


@dataclass(kw_only=True)
class QIf(Statement):
    condition: Expression
    body: Block


@dataclass(kw_only=True)
class While(Statement):
    condition: Expression
    body: Block


@dataclass(kw_only=True)
class LoopVarDecl(Declaration):
    declared_type: Type


@dataclass(kw_only=True)
class For(Statement):
    binding: LoopVarDecl
    iterable: Expression
    body: Block


Symbol = Declaration | ProcParam


@dataclass(kw_only=True)
class Name(Expression):
    name: str
    # A back-reference, not tree structure: traversal must not follow it, or a
    # declaration would be revisited once per reference to it.
    resolved_symbol: Symbol | None = field(default=None, metadata={"annotation": True})


@dataclass(kw_only=True)
class Literal(Expression):
    value: int | float | bool | complex


@dataclass(kw_only=True)
class Index(Expression):
    base: Name
    index: Expression


@dataclass(kw_only=True)
class Call(Expression):
    callee: Name
    args: list[Expression] = field(default_factory=list)


@dataclass(kw_only=True)
class BinaryOp(Expression):
    op: str
    left: Expression
    right: Expression


@dataclass(kw_only=True)
class UnaryOp(Expression):
    op: str
    operand: Expression


__all__ = [
    "AmplitudeList",
    "Assign",
    "AugAssign",
    "BinaryOp",
    "Block",
    "BoolType",
    "BuiltinDecl",
    "Call",
    "ClassicalDecl",
    "ComplexType",
    "Declaration",
    "Expression",
    "ExprStatement",
    "FloatType",
    "For",
    "If",
    "Index",
    "IntType",
    "LoopVarDecl",
    "Literal",
    "Name",
    "Node",
    "ParamArrayDecl",
    "ParamDecl",
    "ProbList",
    "ProcParam",
    "ProcessDef",
    "Program",
    "QBoolType",
    "QIf",
    "QIntType",
    "QuantumDecl",
    "Span",
    "Statement",
    "Symbol",
    "Type",
    "UnaryOp",
    "While",
    "is_quantum",
    "qubit_count",
]
