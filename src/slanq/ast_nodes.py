from __future__ import annotations

from dataclasses import dataclass, field


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
class IntType(Type):
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
class QuantumDecl(Declaration):
    declared_type: Type
    initializer: Expression


@dataclass(kw_only=True)
class ClassicalDecl(Declaration):
    declared_type: Type
    initializer: Expression


@dataclass(kw_only=True)
class ExprStatement(Statement):
    expr: Expression


@dataclass(kw_only=True)
class Name(Expression):
    name: str
    resolved_symbol: Declaration | None = None


@dataclass(kw_only=True)
class Literal(Expression):
    value: int | float | bool


@dataclass(kw_only=True)
class Call(Expression):
    callee: Name
    args: list[Expression] = field(default_factory=list)


__all__ = [
    "Call",
    "ClassicalDecl",
    "Declaration",
    "Expression",
    "ExprStatement",
    "IntType",
    "Literal",
    "Name",
    "Node",
    "Program",
    "QBoolType",
    "QuantumDecl",
    "Span",
    "Statement",
    "Type",
]
