from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import Call, ExprStatement, If, Name, Program, QuantumDecl
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


def test_for_loop_variable_is_in_scope(diagnostics_of: DiagnosticsOf) -> None:
    source = "qbool q = false;\nfor(int i = 0; i < 4; i++) { X(q); }\n"
    assert not diagnostics_of(source).has_errors


def test_process_parameters_are_in_scope(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("process add(qint x, qint y) { x += y; }").has_errors


def test_block_scoped_declaration_is_not_visible_outside(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = "qbool q = false;\nif(1) { qbool inner = false; }\nH(inner);\n"
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert "inner" in bag.errors[0].message


def test_for_loop_variable_is_not_visible_outside(diagnostics_of: DiagnosticsOf) -> None:
    source = "qbool q = false;\nfor(int i = 0; i < 4; i++) { X(q); }\nH(i);\n"
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert "'i'" in bag.errors[0].message


def test_hello_example_resolves_every_name(
    diagnostics_of: DiagnosticsOf, hello_source: str
) -> None:
    """The full-language example must not produce a single undefined-name error."""
    messages = [d.message for d in diagnostics_of(hello_source).errors]
    assert not [m for m in messages if "undefined name" in m]


def test_inner_scope_shadows_outer(analyzed_ast: AnalyzedAst) -> None:
    source = "qbool q = false;\nif(1) { qbool q = true; X(q); }\n"
    ast = analyzed_ast(source)

    outer = ast.statements[0]
    branch = ast.statements[1]
    assert isinstance(outer, QuantumDecl)
    assert isinstance(branch, If)

    inner_decl, gate_stmt = branch.body.statements
    assert isinstance(gate_stmt, ExprStatement)
    assert isinstance(gate_stmt.expr, Call)
    (argument,) = gate_stmt.expr.args
    assert isinstance(argument, Name)
    assert argument.resolved_symbol is inner_decl
    assert argument.resolved_symbol is not outer


def test_index_out_of_range_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> q = 0; H(q[2]);")
    assert bag.has_errors
    assert "out of range" in bag.errors[0].message


def test_index_within_range_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> q = 0; H(q[1]);").has_errors


def test_non_literal_quantum_index_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> q = 0; int i = 0; H(q[i]);")
    assert bag.has_errors
    assert "integer literal" in bag.errors[0].message


def test_classical_array_index_is_not_restricted(diagnostics_of: DiagnosticsOf) -> None:
    """The literal-only rule applies to qubit registers, not classical arrays."""
    source = "param int gamma[4]; for(int i = 0; i < 4; i++) { int x = gamma[i]; }"
    assert not diagnostics_of(source).has_errors


def test_initializer_too_large_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 7;")
    assert bag.has_errors
    assert "does not fit" in bag.errors[0].message


def test_largest_fitting_initializer_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> a = 3;").has_errors
