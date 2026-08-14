from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import Call, ExprStatement, Name, Program, QuantumDecl
from slanq.diagnostics import DiagnosticBag

BuildAst = Callable[[str], Program]
AnalyzedAst = Callable[[str], Program]
DiagnosticsOf = Callable[[str], DiagnosticBag]


@pytest.fixture
def analyzed_ast(build_ast: BuildAst) -> AnalyzedAst:
    def _analyzed_ast(source: str) -> Program:
        ast = build_ast(source)
        analyze(ast, DiagnosticBag())
        return ast

    return _analyzed_ast


@pytest.fixture
def diagnostics_of(build_ast: BuildAst) -> DiagnosticsOf:
    def _diagnostics_of(source: str) -> DiagnosticBag:
        bag = DiagnosticBag()
        analyze(build_ast(source), bag)
        return bag

    return _diagnostics_of


def test_mvp_a_analyzes_clean(diagnostics_of: DiagnosticsOf, mvp_a_source: str) -> None:
    assert not diagnostics_of(mvp_a_source).has_errors


def test_name_resolution_binds_use_to_declaration(analyzed_ast: AnalyzedAst) -> None:
    ast = analyzed_ast("qbool q = false; H(q);")

    declaration = ast.statements[0]
    assert isinstance(declaration, QuantumDecl)

    statement = ast.statements[1]
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)

    (argument,) = statement.expr.args
    assert isinstance(argument, Name)
    assert argument.resolved_symbol is declaration


def test_builtin_callee_has_no_resolved_symbol(analyzed_ast: AnalyzedAst) -> None:
    ast = analyzed_ast("qbool q = false; H(q);")
    statement = ast.statements[1]
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    assert statement.expr.callee.name == "H"
    assert statement.expr.callee.resolved_symbol is None


def test_undefined_name_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("H(unknown);")
    assert bag.has_errors
    assert "unknown" in bag.errors[0].message


def test_builtin_gate_name_is_not_reported(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qbool q = false; H(q);").has_errors


@pytest.mark.parametrize("gate", ["H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg", "SWAP", "CCX"])
def test_builtin_gates_resolve(diagnostics_of: DiagnosticsOf, gate: str) -> None:
    assert not diagnostics_of(f"qbool q = false; {gate}(q);").has_errors


def test_duplicate_declaration_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; qbool q = true;")
    assert bag.has_errors
    assert "q" in bag.errors[0].message


def test_error_carries_source_position(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("H(unknown);")
    assert bag.errors[0].line == 1
    assert bag.errors[0].column is not None
