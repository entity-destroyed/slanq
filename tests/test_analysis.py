from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import (
    BuiltinDecl,
    Call,
    ClassicalDecl,
    ExprStatement,
    If,
    IntType,
    Name,
    Program,
    QuantumDecl,
)
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


def test_builtin_callee_resolves_to_a_builtin(analyzed_ast: AnalyzedAst) -> None:
    ast = analyzed_ast("qbool q = false; H(q);")
    statement = ast.statements[1]
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    symbol = statement.expr.callee.resolved_symbol
    assert isinstance(symbol, BuiltinDecl)
    assert symbol.signature is not None


def test_undefined_name_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("H(unknown);")
    assert bag.has_errors
    assert "unknown" in bag.errors[0].message


def test_builtin_gate_name_is_not_reported(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qbool q = false; H(q);").has_errors


@pytest.mark.parametrize(
    "call",
    [
        "H(q)", "X(q)", "Y(q)", "Z(q)", "S(q)", "T(q)", "Sdg(q)", "Tdg(q)",
        "RX(PI, q)", "RY(PI, q)", "RZ(PI, q)",
        "CX(q, r)", "CY(q, r)", "CZ(q, r)", "CH(q, r)", "SWAP(q, r)",
        "CCX(q, r, s)",
    ],
)
def test_builtin_gates_resolve(diagnostics_of: DiagnosticsOf, call: str) -> None:
    source = f"qbool q = false; qbool r = false; qbool s = false; {call};"
    assert not diagnostics_of(source).has_errors


def test_duplicate_declaration_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; qbool q = true;")
    assert bag.has_errors
    assert "q" in bag.errors[0].message


def test_error_carries_source_position(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("H(unknown);")
    assert bag.errors[0].line == 1
    assert bag.errors[0].column is not None


def test_for_loop_variable_is_in_scope(diagnostics_of: DiagnosticsOf) -> None:
    source = "qbool q = false;\nfor(int i in range(4)) { X(q); }\n"
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
    source = "qbool q = false;\nfor(int i in range(4)) { X(q); }\nH(i);\n"
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


def test_non_constant_quantum_index_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> q = 0; int i = 0; H(q[i]);")
    assert bag.has_errors
    assert "compile-time integer" in bag.errors[0].message


def test_constant_expression_index_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<4> q = 0; H(q[1 + 2]);").has_errors


def test_constant_expression_index_is_range_checked(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<4> q = 0; H(q[2 * 2]);")
    assert bag.has_errors
    assert "out of range" in bag.errors[0].message


def test_fractional_index_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    """`/` yields a float, so `floor()` is needed to get back to an index."""
    bag = diagnostics_of("qint<4> q = 0; H(q[3 / 2]);")
    assert bag.has_errors
    assert "compile-time integer" in bag.errors[0].message
    assert not diagnostics_of("qint<4> q = 0; H(q[floor(3 / 2)]);").has_errors


def test_classical_array_index_is_not_restricted(diagnostics_of: DiagnosticsOf) -> None:
    """The literal-only rule applies to qubit registers, not classical arrays."""
    source = "param int gamma[4]; for(int i in range(4)) { int x = gamma[i]; }"
    assert not diagnostics_of(source).has_errors


def test_initializer_too_large_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 7;")
    assert bag.has_errors
    assert "does not fit" in bag.errors[0].message


def test_largest_fitting_initializer_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> a = 3;").has_errors


def test_for_loop_must_iterate_over_range(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; for(int i in 4) { X(q); }")
    assert bag.has_errors
    assert "range(...)" in bag.errors[0].message


def test_range_argument_count_is_checked(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; for(int i in range(1, 2, 3, 4)) { X(q); }")
    assert bag.has_errors
    assert "range()" in bag.errors[0].message


def test_range_step_may_not_be_zero(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; for(int i in range(0, 4, 0)) { X(q); }")
    assert bag.has_errors
    assert "must not be zero" in bag.errors[0].message


def test_range_forms_are_accepted(diagnostics_of: DiagnosticsOf) -> None:
    for iterable in ("range(4)", "range(1, 4)", "range(0, 8, 2)"):
        source = f"qbool q = false; for(int i in {iterable}) {{ X(q); }}"
        assert not diagnostics_of(source).has_errors, iterable


def test_index_inside_a_loop_is_not_reported(diagnostics_of: DiagnosticsOf) -> None:
    """The loop is not lowered yet, so its variable has no compile-time value.
    Reporting the index would hide the honest 'for loop is not implemented'."""
    source = "qint<4> q = 0; for(int i in range(4)) { X(q[i]); }"
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize(
    ("call", "fragment"),
    [
        ("H(q, q)", "takes 1 argument"),
        ("CX(q)", "takes 2 argument"),
        ("H()", "takes 1 argument"),
        ("RX(q)", "takes 2 argument"),
        ("H(1.5)", "expects a quantum variable"),
        ("CX(1.5, 2.5)", "expects a quantum variable"),
        ("RX(q, PI)", "expects a number here"),
    ],
)
def test_gate_misuse_is_reported(
    diagnostics_of: DiagnosticsOf, call: str, fragment: str
) -> None:
    bag = diagnostics_of(f"qbool q = false; {call};")
    assert bag.has_errors
    assert fragment in bag.errors[0].message


def test_registers_of_different_sizes_are_rejected(diagnostics_of: DiagnosticsOf) -> None:
    """Qiskit has no pairing for these either; it fails when the circuit is built."""
    bag = diagnostics_of("qint<2> a = 0; qint<3> b = 0; CX(a, b);")
    assert bag.has_errors
    assert "sizes 2, 3" in bag.errors[0].message


@pytest.mark.parametrize("call", ["CX(a, b)", "CX(a[0], b)", "CX(a, b[1])", "H(a)"])
def test_broadcast_shapes_are_accepted(diagnostics_of: DiagnosticsOf, call: str) -> None:
    """A single bit spreads over a register; equal sizes pair up."""
    assert not diagnostics_of(f"qint<2> a = 0; qint<2> b = 0; {call};").has_errors


def test_three_distinct_registers_of_equal_size_are_accepted(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """Equal-size registers pair up bitwise even across three arguments, as long
    as they are distinct registers."""
    source = "qint<2> a = 0; qint<2> b = 0; qint<2> c = 0; CCX(a, b, c);"
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize(
    "call",
    [
        "CX(a, a)",
        "CX(a[0], a[0])",
        "CCX(a, b, a)",
        "CCX(a[0], b[0], a[0])",
        "SWAP(a[0], a[0])",
    ],
)
def test_aliased_qubit_arguments_are_rejected(
    diagnostics_of: DiagnosticsOf, call: str
) -> None:
    """Qiskit raises 'duplicate bit arguments' at circuit-build time if the same
    physical qubit is passed twice into one gate call; caught here instead."""
    bag = diagnostics_of(f"qint<2> a = 0; qint<2> b = 0; {call};")
    assert bag.has_errors
    assert "same qubit" in bag.errors[0].message


def test_measure_rejects_a_single_qubit(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> q = 0; int r = measure(q[0]);")
    assert bag.has_errors
    assert "whole quantum variable" in bag.errors[0].message


def test_builtin_name_cannot_be_redeclared(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("int H = 3;")
    assert bag.has_errors
    assert "built into the language" in bag.errors[0].message


def test_gate_used_as_a_value_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; int x = H(q);")
    assert bag.has_errors
    assert "does not return a value" in bag.errors[0].message


def test_measure_result_must_be_used(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qbool q = false; measure(q);")
    assert bag.has_errors
    assert "must be assigned" in bag.errors[0].message


def test_range_outside_a_loop_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("int r = range(4);")
    assert bag.has_errors
    assert "iterable of a for loop" in bag.errors[0].message


def test_call_carries_its_return_type(analyzed_ast: AnalyzedAst) -> None:
    ast = analyzed_ast("qbool q = false; int r = measure(q);")
    declaration = ast.statements[1]
    assert isinstance(declaration, ClassicalDecl)
    assert isinstance(declaration.initializer.inferred_type, IntType)


def test_empty_probability_list_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> a = [];").has_errors


def test_probability_list_length_must_match_two_to_the_size(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = [0.5, 0.5];")
    assert bag.has_errors
    assert "needs 4 probabilities (2^2), got 2" in bag.errors[0].message


def test_probability_over_one_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = [1.5, 0.1, 0.5, 0.5];")
    assert bag.has_errors
    assert "out of range" in bag.errors[0].message


def test_probabilities_summing_to_zero_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = [0, 0, 0, 0];")
    assert bag.has_errors
    assert "cannot sum to zero" in bag.errors[0].message


def test_probabilities_off_by_a_meaningful_amount_warn(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = [0.1, 0.1, 0.1, 0.1];")
    assert not bag.has_errors
    assert bag.warnings
    assert "sum to 0.4" in bag.warnings[0].message


def test_floating_point_noise_does_not_warn(diagnostics_of: DiagnosticsOf) -> None:
    """0.15 + 0.15 + 0.35 + 0.35 sums to 0.9999999999999999 in floating point;
    an obviously-intended list like this stays silent."""
    bag = diagnostics_of("qint<2> a = [0.15, 0.15, 0.35, 0.35];")
    assert not bag.has_errors
    assert not bag.warnings


def test_qbool_probability_list_needs_exactly_two_entries(
    diagnostics_of: DiagnosticsOf,
) -> None:
    assert not diagnostics_of("qbool a = [0.3, 0.7];").has_errors
    bag = diagnostics_of("qbool a = [0.3, 0.3, 0.4];")
    assert bag.has_errors
    assert "needs 2 probabilities" in bag.errors[0].message


def test_empty_amplitude_list_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> a = {};").has_errors


def test_amplitude_list_length_must_match_two_to_the_size(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = {0.5, 0.5};")
    assert bag.has_errors
    assert "needs 4 amplitudes (2^2), got 2" in bag.errors[0].message


def test_amplitude_list_accepts_negative_and_complex_values(
    diagnostics_of: DiagnosticsOf,
) -> None:
    assert not diagnostics_of("qint<1> a = {0.6, -0.8};").has_errors
    assert not diagnostics_of("qint<1> a = {1/sqrt(2), 1i/sqrt(2)};").has_errors


def test_amplitude_list_rejects_a_non_constant_element(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("int x = 5; qint<1> a = {x, 0};")
    assert bag.has_errors
    assert "not a compile-time constant" in bag.errors[0].message


def test_amplitude_list_rejects_a_boolean_element(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<1> a = {true, false};")
    assert bag.has_errors
    assert "cannot be a boolean" in bag.errors[0].message


def test_amplitudes_summing_to_zero_norm_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<1> a = {0, 0};")
    assert bag.has_errors
    assert "cannot all be zero" in bag.errors[0].message


def test_amplitudes_off_by_a_meaningful_amount_warn(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<1> a = {0.6, 0.6};")
    assert not bag.has_errors
    assert bag.warnings
    assert "squared norm" in bag.warnings[0].message


def test_amplitude_list_unnormalized_but_within_tolerance_is_silent(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """0.6**2 + 0.8**2 is exactly 1.0; no floating-point noise to tolerate here,
    but this mirrors the analogous [] floating-point-noise test in spirit."""
    bag = diagnostics_of("qint<1> a = {0.6, 0.8};")
    assert not bag.has_errors
    assert not bag.warnings


def test_qif_equality_condition_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a == 2) { X(out); }"
    ).has_errors


def test_qif_bare_qubit_condition_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of(
        "qbool flag = false; qbool out = false; qif(flag) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a[0]) { X(out); }"
    ).has_errors


def test_qif_and_chain_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    source = (
        "qint<2> a = 0; qbool flag = false; qbool out = false; "
        "qif(a == 2 && flag) { X(out); }"
    )
    assert not diagnostics_of(source).has_errors


def test_qif_leaf_and_top_level_negation_are_accepted(
    diagnostics_of: DiagnosticsOf,
) -> None:
    assert not diagnostics_of(
        "qbool flag = false; qbool out = false; qif(!flag) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(!(a == 2)) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a[0] && !a[1]) { X(out); }"
    ).has_errors


def test_qif_classical_condition_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of(
        "int r = 5; qbool out = false; qif(r == 5) { X(out); }"
    )
    assert bag.has_errors
    assert "may only reference quantum variables" in bag.errors[0].message


def test_qif_mixed_classical_and_quantum_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """One unified rule, not two: every clause is checked on its own, so a
    mix is caught by the exact same check that catches a purely classical one."""
    bag = diagnostics_of(
        "int r = 5; qint<2> a = 0; qbool out = false; "
        "qif(r == 5 && a == 2) { X(out); }"
    )
    assert bag.has_errors
    assert "may only reference quantum variables" in bag.errors[0].message


def test_qif_or_is_not_implemented(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a == 2 || a == 1) { X(out); }"
    )
    assert bag.has_errors
    assert "'||'" in bag.errors[0].message


def test_qif_inequality_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    """`!=` is sugar for a negated equality test -- same rules as `!(a==c)`."""
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a != 2) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(2 != a) { X(out); }"
    ).has_errors


def test_qif_equality_accepts_either_operand_order(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(2 == a) { X(out); }"
    ).has_errors


def test_qif_deeply_nested_negation_is_not_implemented(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = (
        "qint<2> a = 0; qbool flag = false; qbool out = false; "
        "qif(flag && !(a[0] && a[1])) { X(out); }"
    )
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert "deeper nesting" in bag.errors[0].message


def test_qif_bare_multi_qubit_register_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    """A bare condition must be exactly one qubit; `a` alone isn't sugar for
    anything well-defined when a is wider than that."""
    bag = diagnostics_of("qint<2> a = 0; qbool out = false; qif(a) { X(out); }")
    assert bag.has_errors
    assert "exactly one" in bag.errors[0].message


def test_qif_equality_constant_out_of_range_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = 0; qbool out = false; qif(a == 7) { X(out); }")
    assert bag.has_errors
    assert "does not fit" in bag.errors[0].message


def test_qif_body_may_not_touch_the_condition_qubits(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = 0; qif(a == 2) { X(a[0]); }")
    assert bag.has_errors
    assert "may not modify" in bag.errors[0].message


def test_qif_body_may_touch_a_different_variable(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a == 2) { X(out); }"
    ).has_errors


def test_qif_negated_wide_equality_combines_with_and(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """A wide negated clause gets its own ancilla to compute; combining it
    with other clauses via '&&' does not need OR-like machinery -- only
    negating a whole compound sub-expression (De Morgan) does."""
    assert not diagnostics_of(
        "qint<2> a = 0; qbool flag = false; qbool out = false; "
        "qif(flag && !(a == 2)) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qint<2> b = 0; qbool out = false; "
        "qif(!(a == 2) && !(b == 3)) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a != 2) { X(out); }"
    ).has_errors


def test_qif_negated_single_qubit_equality_combines_fine(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = (
        "qint<1> q = 0; qbool flag = false; qbool out = false; "
        "qif(flag && !(q == 1)) { X(out); }"
    )
    assert not diagnostics_of(source).has_errors


def test_augassign_quantum_addend_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> a = 0; qint<2> b = 0; a += b;").has_errors
    assert not diagnostics_of("qint<3> a = 0; qint<2> b = 0; a += b;").has_errors
    assert not diagnostics_of("qint<2> a = 0; qint<3> b = 0; a -= b;").has_errors


def test_augassign_constant_addend_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> a = 0; a += 3;").has_errors


def test_augassign_indexed_target_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; qbool b = false; a[0] += b;")
    assert bag.has_errors
    assert "not a single qubit" in bag.errors[0].message


def test_augassign_classical_target_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("int x = 0; x += 1;")
    assert bag.has_errors
    assert "is not a quantum variable" in bag.errors[0].message


def test_augassign_param_addend_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; param int gamma; a += gamma;")
    assert bag.has_errors
    assert "runtime parameter" in bag.errors[0].message


def test_augassign_param_array_addend_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; param int gamma[2]; a += gamma[0];")
    assert bag.has_errors
    assert "runtime parameter" in bag.errors[0].message


def test_augassign_multiply_of_two_quantum_variables_is_accepted(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = "qint<2> a = 0; qint<2> b = 0; qint<2> c = 0; a += b * c;"
    assert not diagnostics_of(source).has_errors


def test_augassign_multiply_with_a_classical_operand_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = 0; qint<2> b = 0; int x = 0; a += b * x;")
    assert bag.has_errors
    assert "'x' is not a quantum variable" in bag.errors[0].message


def test_augassign_unsupported_shape_is_reported(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; a += 1.5;")
    assert bag.has_errors
    assert "not implemented yet" in bag.errors[0].message


def test_augassign_self_addend_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; a += a;")
    assert bag.has_errors
    assert "cannot appear on both sides" in bag.errors[0].message


def test_augassign_self_subtract_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; a -= a;")
    assert bag.has_errors
    assert "cannot appear on both sides" in bag.errors[0].message


@pytest.mark.parametrize(
    "source",
    [
        "qint<2> a = 0; qint<2> b = 0; a += a * b;",
        "qint<2> a = 0; qint<2> b = 0; a += b * a;",
        "qint<2> a = 0; qint<2> b = 0; b += a * b;",
    ],
)
def test_augassign_multiply_aliasing_the_target_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert "cannot appear on both sides" in bag.errors[0].message


def test_augassign_multiply_with_a_non_name_operand_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = 0; qint<2> b = 0; a += b * 2;")
    assert bag.has_errors
    assert "must multiply two whole quantum variables" in bag.errors[0].message


def test_quantum_decl_multiply_initializer_is_accepted(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = "qint<2> a = 0; qint<2> b = 0; qint<4> c = a * b;"
    assert not diagnostics_of(source).has_errors


def test_quantum_decl_multiply_with_a_classical_operand_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = 0; int x = 0; qint<4> c = a * x;")
    assert bag.has_errors
    assert "'x' is not a quantum variable" in bag.errors[0].message


def test_param_complex_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    """No Qiskit gate-synthesis primitive ever consumes a complex-valued
    Parameter -- measured against StatePreparation, UnitaryGate and
    PauliEvolutionGate, all reject one. A param complex would never have a
    working consumer."""
    bag = diagnostics_of("param complex theta;")
    assert bag.has_errors
    assert "complex is not supported" in bag.errors[0].message


def test_param_complex_array_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("param complex gamma[3];")
    assert bag.has_errors
    assert "complex is not supported" in bag.errors[0].message
