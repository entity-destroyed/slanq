from __future__ import annotations

from collections.abc import Callable

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
)

BuildAst = Callable[[str], Program]


def test_qbool_decl_becomes_quantum_decl(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qbool q = false;").statements
    assert isinstance(declaration, QuantumDecl)
    assert declaration.name == "q"
    assert isinstance(declaration.declared_type, QBoolType)
    assert isinstance(declaration.initializer, Literal)
    assert declaration.initializer.value is False


def test_gate_call_becomes_expr_statement(build_ast: BuildAst) -> None:
    (statement,) = build_ast("H(q);").statements
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    assert statement.expr.callee.name == "H"
    (argument,) = statement.expr.args
    assert isinstance(argument, Name)
    assert argument.name == "q"


def test_call_without_arguments_has_empty_arg_list(build_ast: BuildAst) -> None:
    (statement,) = build_ast("H();").statements
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    assert statement.expr.args == []


def test_classical_decl_with_measurement(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("int result = measure(q);").statements
    assert isinstance(declaration, ClassicalDecl)
    assert declaration.name == "result"
    assert isinstance(declaration.declared_type, IntType)
    assert isinstance(declaration.initializer, Call)
    assert declaration.initializer.callee.name == "measure"


def test_mvp_a_full_source(build_ast: BuildAst, mvp_a_source: str) -> None:
    statements = build_ast(mvp_a_source).statements
    assert len(statements) == 3
    assert isinstance(statements[0], QuantumDecl)
    assert isinstance(statements[1], ExprStatement)
    assert isinstance(statements[2], ClassicalDecl)


def test_spans_are_populated(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qbool q = false;").statements
    assert declaration.span.start_line == 1
    assert declaration.span.start_col >= 1


def test_number_literal_stays_an_int(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("int x = 42;").statements
    assert isinstance(declaration, ClassicalDecl)
    assert isinstance(declaration.initializer, Literal)
    assert declaration.initializer.value == 42
    assert isinstance(declaration.initializer.value, int)


def test_boolean_true_literal(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qbool q = true;").statements
    assert isinstance(declaration, QuantumDecl)
    assert isinstance(declaration.initializer, Literal)
    assert declaration.initializer.value is True
