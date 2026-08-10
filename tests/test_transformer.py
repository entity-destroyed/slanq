from __future__ import annotations

from pathlib import Path

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
from slanq.parser import parse_source
from slanq.transformer import transform_to_ast

EXAMPLES = Path(__file__).parent.parent / "examples"


def _ast(source: str) -> Program:
    return transform_to_ast(parse_source(source))


def test_qbool_decl_becomes_quantum_decl() -> None:
    ast = _ast("qbool q = false;")
    assert isinstance(ast, Program)
    (decl,) = ast.statements
    assert isinstance(decl, QuantumDecl)
    assert decl.name == "q"
    assert isinstance(decl.declared_type, QBoolType)
    assert isinstance(decl.initializer, Literal)
    assert decl.initializer.value is False


def test_gate_call_becomes_expr_statement() -> None:
    ast = _ast("H(q);")
    (stmt,) = ast.statements
    assert isinstance(stmt, ExprStatement)
    assert isinstance(stmt.expr, Call)
    assert stmt.expr.callee.name == "H"
    (arg,) = stmt.expr.args
    assert isinstance(arg, Name)
    assert arg.name == "q"


def test_classical_decl_with_measurement() -> None:
    ast = _ast("int result = measure(q);")
    (decl,) = ast.statements
    assert isinstance(decl, ClassicalDecl)
    assert decl.name == "result"
    assert isinstance(decl.declared_type, IntType)
    assert isinstance(decl.initializer, Call)
    assert decl.initializer.callee.name == "measure"


def test_mvp_a_full_source() -> None:
    source = (EXAMPLES / "mvp_a.slanq").read_text(encoding="utf-8")
    ast = _ast(source)
    assert len(ast.statements) == 3
    assert isinstance(ast.statements[0], QuantumDecl)
    assert isinstance(ast.statements[1], ExprStatement)
    assert isinstance(ast.statements[2], ClassicalDecl)


def test_spans_are_populated() -> None:
    ast = _ast("qbool q = false;")
    decl = ast.statements[0]
    assert decl.span.start_line == 1
    assert decl.span.start_col >= 1


def test_number_literal_int() -> None:
    ast = _ast("int x = 42;")
    decl = ast.statements[0]
    assert isinstance(decl, ClassicalDecl)
    assert isinstance(decl.initializer, Literal)
    assert decl.initializer.value == 42
    assert isinstance(decl.initializer.value, int)


def test_boolean_true_literal() -> None:
    ast = _ast("qbool q = true;")
    decl = ast.statements[0]
    assert isinstance(decl, QuantumDecl)
    assert isinstance(decl.initializer, Literal)
    assert decl.initializer.value is True
