from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import (
    BuiltinDecl,
    Call,
    ExprStatement,
    If,
    IntType,
    Name,
    Program,
    QuantumDecl,
    RealtimeDecl,
)
from slanq.diagnostics import DiagnosticBag
from slanq.parser import reserved_words

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
        "H(q)",
        "X(q)",
        "Y(q)",
        "Z(q)",
        "S(q)",
        "T(q)",
        "Sdg(q)",
        "Tdg(q)",
        "RX(PI, q)",
        "RY(PI, q)",
        "RZ(PI, q)",
        "CX(q, r)",
        "CY(q, r)",
        "CZ(q, r)",
        "CH(q, r)",
        "SWAP(q, r)",
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


def test_process_call_arity_is_checked(diagnostics_of: DiagnosticsOf) -> None:
    source = "process add(qint x, qint y) { x += y; }\nqint<2> a = 1;\nadd(a);\n"
    bag = diagnostics_of(source)
    assert any("takes 2 argument(s), got 1" in d.message for d in bag.errors)


def test_process_call_rejects_the_same_variable_twice(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """The section 3.4 aliasing ban. A call-site rule, so it is checked before
    expansion -- afterwards there is no call left to point at."""
    source = "process add(qint x, qint y) { x += y; }\nqint<2> a = 1;\nadd(a, a);\n"
    bag = diagnostics_of(source)
    assert any("same qubit(s) in more than one argument" in d.message for d in bag.errors)


def test_process_call_rejects_the_same_qubit_twice(diagnostics_of: DiagnosticsOf) -> None:
    """Aliasing is about physical qubits, not names: two different spellings
    of the same bit alias just as much."""
    source = "process pair(qbool p, qbool r) { CX(p, r); }\nqint<2> a = 0;\npair(a[0], a[0]);\n"
    bag = diagnostics_of(source)
    assert any("same qubit(s) in more than one argument" in d.message for d in bag.errors)


def test_two_distinct_qubits_of_one_register_are_accepted(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = "process pair(qbool p, qbool r) { CX(p, r); }\nqint<2> a = 0;\npair(a[0], a[1]);\n"
    assert not diagnostics_of(source).has_errors


def test_a_process_call_must_be_a_statement(diagnostics_of: DiagnosticsOf) -> None:
    source = "process f(qint x) { X(x); }\nqint<2> a = 0;\nint r = f(a);\n"
    bag = diagnostics_of(source)
    assert any("returns no value" in d.message for d in bag.errors)


def test_calling_a_variable_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    """Previously nothing checked this: a callee resolving to neither a
    builtin nor a process fell through analysis silently."""
    bag = diagnostics_of("qint<2> a = 0;\na(1);\n")
    assert any("not a gate, a function or a process" in d.message for d in bag.errors)


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
    """A genuinely runtime value (a measurement result) stays rejected -- a
    classical constant, tested just below, is a different case now."""
    bag = diagnostics_of("qint<2> q = 0; qbool flag = false; rt int<> i = measure(flag); H(q[i]);")
    assert bag.has_errors
    assert "an integer the compiler can compute" in bag.errors[0].message


def test_classical_int_name_index_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qint<2> q = 0; int i = 0; H(q[i]);").has_errors


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
    assert "an integer the compiler can compute" in bag.errors[0].message
    assert not diagnostics_of("qint<4> q = 0; H(q[floor(3 / 2)]);").has_errors


def test_a_param_array_index_within_bounds_is_accepted(
    diagnostics_of: DiagnosticsOf,
) -> None:
    loop = "param float[4] gamma; qbool q = false; for(int i in range(4)) { RX(gamma[i], q); }"
    assert not diagnostics_of(loop).has_errors


@pytest.mark.parametrize(
    "source",
    [
        "param int[4] gamma; int x = gamma[99];",
        "param float[2] gamma; qbool q = false; RX(gamma[2], q);",
        "param float[2] gamma; qbool q = false; RX(gamma[-1], q);",
        "param float[3] gamma; qbool q = false; for(int i in range(4)) { RX(gamma[i], q); }",
    ],
)
def test_a_param_array_index_out_of_bounds_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    """A param array reaches the generated file as a Python list, so an index
    past its end raises IndexError there -- on the user's machine, long after
    the compiler said yes. Unrolling has made every index a constant by this
    point, so the check is complete."""
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert "is out of range for 'gamma'" in bag.errors[0].message


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("qint<2> a = 2; qint<4> c = a * a;", "for both operands"),
        ("qint<2> a = 2; qint<4> c = 0; c += a * a;", "for both operands"),
        (
            "qint<2> a = 2; qint<2> b = 3; qint<1> c = a * b;",
            "at least as many qubits as its widest operand",
        ),
        (
            "qint<3> a = 2; qint<2> b = 3; qint<2> c = a * b;",
            "at least as many qubits as its widest operand",
        ),
    ],
)
def test_a_product_the_multiplier_cannot_take_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert expected in bag.errors[0].message


@pytest.mark.parametrize(
    "source",
    [
        "qint<2> a = 2; qint<2> b = 3; qint<2> c = a * b;",
        "qint<2> a = 2; qint<2> b = 3; qint<4> c = a * b;",
        "qint<3> a = 2; qint<2> b = 3; qint<3> c = a * b;",
        "qint<2> a = 2; qint<2> b = 3; qint<1> c = 0; c += a * b;",
    ],
)
def test_a_product_at_least_as_wide_as_its_operands_is_accepted(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    """The width rule is the declaration's, not the `+=`'s: a `c += a * b`
    computes the product at full width in a temporary first, so any target
    width works there and truncates mod 2^size like every other assignment."""
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("qint<0> a = 0;", "needs at least one qubit"),
        ("param int[0] g;", "needs at least one element"),
    ],
)
def test_a_declared_size_of_zero_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """A register with no qubits holds no value, and an empty param array has
    no index that could be in range -- both parse, and neither means
    anything."""
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert expected in bag.errors[0].message


@pytest.mark.parametrize("source", ["qint<1> a = 1;", "param int[1] g;"])
def test_a_declared_size_of_one_is_accepted(diagnostics_of: DiagnosticsOf, source: str) -> None:
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("qint<2.5> a = 0;", "a width must be a whole number, not a fraction"),
        ("qint<true> a = 0;", "a width must be a whole number, not a boolean"),
        ("qint<1 + 1i> a = 0;", "a width must be a whole number, not a complex number"),
        ("param int[0.5] g;", "a length must be a whole number, not a fraction"),
        (
            "qint<2> a = 0;\nrt int<2.5> r = measure(a);",
            "a width must be a whole number, not a fraction",
        ),
        (
            "qint<2> a = 0;\nrt bool[2.5] m = measure(a);",
            "a length must be a whole number, not a fraction",
        ),
    ],
)
def test_a_width_that_is_not_a_whole_number_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    assert [error.message for error in diagnostics_of(source).errors] == [expected]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "param float t;\nqint<t> a = 0;",
            "a width must be a whole number the compiler can compute",
        ),
        (
            "param float t;\nparam int[t] g;",
            "a length must be a whole number the compiler can compute",
        ),
    ],
)
def test_a_width_the_compiler_cannot_compute_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """A `param` has no value until the circuit is bound, and a register's
    width has to be known while the circuit is built."""
    assert [error.message for error in diagnostics_of(source).errors] == [expected]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("qint<-1> a = 0;", "needs at least one qubit"),
        ("param int[-1] g;", "needs at least one element"),
    ],
)
def test_a_negative_declared_size_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """A width is an expression now, so a negative one reaches the same check
    an explicit zero does rather than failing to parse."""
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert expected in bag.errors[0].message


@pytest.mark.parametrize(
    "source",
    [
        "int k = 2;\nqint<k * 2> a = 0;\nX(a[3]);",
        "int k = 3;\nqint<3> a = 0;\nrt int<k> r = measure(a);",
        "int k = 2;\nparam int[k + 1] g;\nqbool q = false;\nRX(g[2], q);",
    ],
)
def test_a_computed_width_is_accepted(diagnostics_of: DiagnosticsOf, source: str) -> None:
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize("source", ["qint<2> a = true;", "qint<1> a = false;"])
def test_a_boolean_does_not_initialize_a_qint(diagnostics_of: DiagnosticsOf, source: str) -> None:
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert "holds a number, not a boolean" in bag.errors[0].message


@pytest.mark.parametrize("source", ["qbool q = 1;", "qbool q = 0;", "qbool q = true;"])
def test_a_qbool_takes_either_spelling_of_its_two_values(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    """The reverse of the rule above is not a type confusion: 0 and 1 are
    exactly what one qubit holds."""
    assert not diagnostics_of(source).has_errors


def test_an_empty_program_is_accepted_with_a_warning(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("")
    assert not bag.has_errors
    assert "this program is empty" in bag.warnings[0].message


@pytest.mark.parametrize("source", ["qbool ψ = false;", "qint<3> φ = 0;", "int θ = 1;"])
def test_a_name_with_letters_beyond_ascii_is_accepted(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    assert not diagnostics_of(source).has_errors


def test_a_name_python_would_rewrite_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    """Python normalizes identifiers to NFKC, so `\ufb01x` would become `fix` in the
    generated file -- and could silently merge with a different Slanq name."""
    bag = diagnostics_of("qbool \ufb01x = false;")
    assert bag.has_errors
    assert "would not reach the generated file unchanged" in bag.errors[0].message


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
def test_gate_misuse_is_reported(diagnostics_of: DiagnosticsOf, call: str, fragment: str) -> None:
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
def test_aliased_qubit_arguments_are_rejected(diagnostics_of: DiagnosticsOf, call: str) -> None:
    """Qiskit raises 'duplicate bit arguments' at circuit-build time if the same
    physical qubit is passed twice into one gate call; caught here instead."""
    bag = diagnostics_of(f"qint<2> a = 0; qint<2> b = 0; {call};")
    assert bag.has_errors
    assert "same qubit" in bag.errors[0].message


def test_measure_rejects_a_single_qubit(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> q = 0; rt int<> r = measure(q[0]);")
    assert bag.has_errors
    assert "whole quantum variable" in bag.errors[0].message


def test_builtin_name_cannot_be_redeclared(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("int H = 3;")
    assert bag.has_errors
    assert "built into the language" in bag.errors[0].message


def test_python_keyword_cannot_be_used_as_a_name(diagnostics_of: DiagnosticsOf) -> None:
    """A Python keyword becomes a bare identifier in the generated file
    (`class = QuantumRegister(...)`), which is a SyntaxError -- caught here
    instead of surfacing as an unreadable generated-file failure."""
    bag = diagnostics_of("qbool class = false;")
    assert bag.has_errors
    assert "reserved Python keyword" in bag.errors[0].message


def test_codegen_internal_name_cannot_be_used_as_a_name(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """`circuit` is the compiler's own generated variable name for the
    QuantumCircuit object -- reusing it would shadow it in the generated
    file (measured: crashes with IndexError at generated-file runtime)."""
    bag = diagnostics_of("qbool circuit = false;")
    assert bag.has_errors
    assert "reserved for the compiler" in bag.errors[0].message


def test_reserved_name_check_covers_nested_declarations(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """The same check applies wherever a name is declared, not just at the
    top level: a process parameter and a for-loop variable."""
    bag = diagnostics_of("process p(qint class) { X(class); }")
    assert bag.has_errors
    assert any("reserved Python keyword" in error.message for error in bag.errors)

    bag = diagnostics_of("for(int circuit in range(3)) { int x = circuit; }")
    assert bag.has_errors
    assert any("reserved for the compiler" in error.message for error in bag.errors)


@pytest.mark.parametrize(
    "source",
    [
        "qbool _x = false;",
        "qint<2> _x = 0;",
        "int _x = 1;",
        "param float _x;",
        "param int[2] _x;",
        "process _p(qint x) { X(x); }",
        "process p(qint _x) { X(_x); }",
        "qint<2> a = 0; for(int _i in range(2)) { X(a[_i]); }",
        "qbool _ = false;",
    ],
)
def test_a_name_starting_with_an_underscore_is_rejected(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    """The compiler's own generated names all start with an underscore, and
    there are more of them every time the generator grows (`_ancilla_N`,
    `_qif_body_N`, `_round`, `_sqrt`). Reserving the shape rather than listing
    the names is what makes that list impossible to fall behind -- it had
    already fallen behind on `_sqrt`. The rule covers every kind of
    declaration, including the ones substituted away before code generation,
    so there is no exception to remember."""
    bag = diagnostics_of(source)
    assert bag.has_errors
    assert any("starts with '_'" in error.message for error in bag.errors)


@pytest.mark.parametrize("word", sorted(reserved_words()))
def test_a_keyword_cannot_be_used_as_a_name(diagnostics_of: DiagnosticsOf, word: str) -> None:
    """Parametrised over the derived set rather than a literal list, so a
    keyword added to the grammar is covered here without anyone remembering to
    add it."""
    bag = diagnostics_of(f"int {word} = 3;")
    assert [error.message for error in bag.errors] == [
        f"'{word}' is a keyword of the language and cannot be used as a name"
    ]


@pytest.mark.parametrize(
    "source,word",
    [
        ("process p(qint rt) { X(rt); }", "rt"),
        ("qint<2> a = 0; for(int in in range(2)) { X(a[0]); }", "in"),
        ("param int[2] qif;", "qif"),
    ],
)
def test_the_keyword_rule_covers_nested_declarations(
    diagnostics_of: DiagnosticsOf, source: str, word: str
) -> None:
    bag = diagnostics_of(source)
    assert bag.errors[0].message == (
        f"'{word}' is a keyword of the language and cannot be used as a name"
    )


def test_keywords_are_derived_from_every_terminal_shape() -> None:
    """The grammar spells keywords two ways -- as a bare literal in a rule
    (`qif`) and as an alternation inside a terminal (`CTYPE`, `BOOL`) -- and
    both must contribute. A terminal that matches more than a fixed set of
    words contributes none, which is what keeps every user-chosen name legal."""
    words = reserved_words()
    assert {"qif", "process", "rt"} <= words
    assert {"int", "float", "bool", "complex"} <= words
    assert {"true", "false"} <= words
    assert not {"measure", "PI", "theta", "a"} & words


@pytest.mark.parametrize("source", ["qbool q_ = false;", "qbool a_b = false;", "int x2_ = 1;"])
def test_an_underscore_elsewhere_in_a_name_is_accepted(
    diagnostics_of: DiagnosticsOf, source: str
) -> None:
    assert not diagnostics_of(source).has_errors


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
    ast = analyzed_ast("qbool q = false; rt int<> r = measure(q);")
    declaration = ast.statements[1]
    assert isinstance(declaration, RealtimeDecl)
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
    """A classical constant resolves fine here (see
    test_amplitude_list_accepts_a_classical_name below) -- a `param`, which
    has no compile-time value at all, is what stays rejected."""
    bag = diagnostics_of("param float x; qint<1> a = {x, 0};")
    assert bag.has_errors
    assert "not a compile-time constant" in bag.errors[0].message


def test_amplitude_list_accepts_a_classical_name(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("float x = 0.6; qint<1> a = {x, -0.8};").has_errors


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
    assert not diagnostics_of("qint<2> a = 0; qbool out = false; qif(a[0]) { X(out); }").has_errors


def test_qif_and_chain_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    source = "qint<2> a = 0; qbool flag = false; qbool out = false; qif(a == 2 && flag) { X(out); }"
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
    bag = diagnostics_of("int r = 5; qbool out = false; qif(r == 5) { X(out); }")
    assert bag.has_errors
    assert "may only reference quantum variables" in bag.errors[0].message


def test_qif_mixed_classical_and_quantum_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """One unified rule, not two: every clause is checked on its own, so a
    mix is caught by the exact same check that catches a purely classical one."""
    bag = diagnostics_of(
        "int r = 5; qint<2> a = 0; qbool out = false; qif(r == 5 && a == 2) { X(out); }"
    )
    assert bag.has_errors
    assert "may only reference quantum variables" in bag.errors[0].message


def test_qif_or_is_not_implemented(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; qbool out = false; qif(a == 2 || a == 1) { X(out); }")
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


def test_an_empty_qif_body_is_warned_about(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; qif(a == 2) { }")
    assert not bag.has_errors
    assert "this qif body is empty" in bag.warnings[0].message


def test_a_qif_body_emptied_by_unrolling_is_warned_about(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """The body is written non-empty, so only the unrolled tree shows that
    nothing is left in it."""
    bag = diagnostics_of(
        "qint<2> a = 0; qbool t = false; qif(a == 2) { for(int i in range(0)) { X(t); } }"
    )
    assert not bag.has_errors
    assert any("this qif body is empty" in d.message for d in bag.warnings)


def test_a_qif_body_holding_only_a_phase_is_not_empty(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of("qint<2> a = 0; qif(a == 2) { phase(1.5); }")
    assert not bag.has_errors
    assert not bag.warnings


def test_qif_negated_wide_equality_combines_with_and(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """A wide negated clause gets its own ancilla to compute; combining it
    with other clauses via '&&' does not need OR-like machinery -- only
    negating a whole compound sub-expression (De Morgan) does."""
    assert not diagnostics_of(
        "qint<2> a = 0; qbool flag = false; qbool out = false; qif(flag && !(a == 2)) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qint<2> b = 0; qbool out = false; qif(!(a == 2) && !(b == 3)) { X(out); }"
    ).has_errors
    assert not diagnostics_of(
        "qint<2> a = 0; qbool out = false; qif(a != 2) { X(out); }"
    ).has_errors


def test_qif_negated_single_qubit_equality_combines_fine(
    diagnostics_of: DiagnosticsOf,
) -> None:
    source = (
        "qint<1> q = 0; qbool flag = false; qbool out = false; qif(flag && !(q == 1)) { X(out); }"
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


def test_augassign_on_a_classical_target_is_ordinary_assignment(
    diagnostics_of: DiagnosticsOf,
) -> None:
    assert not diagnostics_of("int x = 0; x += 1;").has_errors


def test_augassign_param_addend_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; param int gamma; a += gamma;")
    assert bag.has_errors
    assert "runtime parameter" in bag.errors[0].message


def test_augassign_param_array_addend_is_rejected(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of("qint<2> a = 0; param int[2] gamma; a += gamma[0];")
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
    bag = diagnostics_of("param complex[3] gamma;")
    assert bag.has_errors
    assert "complex is not supported" in bag.errors[0].message


@pytest.mark.parametrize("declared", ["float", "bool", "complex"])
def test_a_loop_variable_must_be_an_int(diagnostics_of: DiagnosticsOf, declared: str) -> None:
    bag = diagnostics_of(f"qbool q = false;\nfor({declared} i in range(2)) {{ X(q); }}\n")
    assert any("so it is an int" in error.message for error in bag.errors)


def test_an_int_loop_variable_is_accepted(diagnostics_of: DiagnosticsOf) -> None:
    assert not diagnostics_of("qbool q = false;\nfor(int i in range(2)) { X(q); }\n").has_errors


_DECLS = "qint<2> a = 0;\nqbool f = false;\nint n = 5;\nbool t = true;\n"


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("a == 2", "'a' is a quantum variable, so it cannot be an if condition; use qif"),
        ("a[0]", "'a' is a quantum variable, so it cannot be an if condition; use qif"),
        ("n", "an if condition must be a true/false value"),
        ("n + 1", "an if condition must be a true/false value"),
        ("3.5", "an if condition must be a true/false value"),
        ("n / 0 > 1", "division by zero"),
    ],
)
def test_an_if_condition_is_checked(
    diagnostics_of: DiagnosticsOf, condition: str, message: str
) -> None:
    bag = diagnostics_of(f"{_DECLS}if ({condition}) {{ X(f); }}\n")
    (diagnostic,) = bag.errors
    assert diagnostic.message == message


def test_a_param_cannot_be_an_if_condition(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of(f"param int p;\n{_DECLS}if (p > 1) {{ X(f); }}\n")
    (diagnostic,) = bag.errors
    assert diagnostic.message == (
        "'p' is a param, which has no value while the circuit is being built, "
        "so it cannot be an if condition"
    )


def test_a_realtime_value_cannot_be_a_build_time_if_condition(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of(f"{_DECLS}rt int<> r = measure(a);\nif (r < 4) {{ X(f); }}\n")
    (diagnostic,) = bag.errors
    assert diagnostic.message == (
        "'r' is a real-time value, which has no value while the circuit is being built; use rt if"
    )


@pytest.mark.parametrize(
    ("body", "name"),
    [
        ("if (t) { qbool z = false; }", "z"),
        ("if (t) { int k = 1; }", "k"),
        ("if (t) { X(f); } else { qbool z = false; }", "z"),
    ],
    ids=["quantum", "classical", "in the else"],
)
def test_a_declaration_in_an_if_body_is_rejected(
    diagnostics_of: DiagnosticsOf, body: str, name: str
) -> None:
    bag = diagnostics_of(_DECLS + body + "\n")
    (diagnostic,) = bag.errors
    assert diagnostic.message == f"'{name}' cannot be declared inside an if body"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("if (t) { }", "this if body is empty"),
        ("if (t) { X(f); } else { }", "this else body is empty"),
    ],
    ids=["if", "else"],
)
def test_an_empty_branch_warns(diagnostics_of: DiagnosticsOf, body: str, message: str) -> None:
    bag = diagnostics_of(_DECLS + body + "\n")
    assert not bag.has_errors
    (warning,) = bag.warnings
    assert warning.message == message


@pytest.mark.parametrize(
    "condition",
    ["t", "!t", "n > 3", "n > 1 && n < 9", "n > 9 || t", "floor(n / 2) == 2"],
)
def test_a_boolean_condition_is_accepted(diagnostics_of: DiagnosticsOf, condition: str) -> None:
    bag = diagnostics_of(f"{_DECLS}if ({condition}) {{ X(f); }}\n")
    assert not bag.has_errors


_RT_DECLS = (
    "qint<3> a = [];\nqint<2> b = 0;\nqbool f = false;\nint n = 5;\nbool t = true;\n"
    "rt int<> m = measure(a);\n"
)


@pytest.mark.parametrize(
    ("program", "message"),
    [
        (
            "rt if (b == 2) { X(f); }",
            "'b' is a quantum variable, so it cannot be an rt if condition; use qif",
        ),
        (
            "rt if (n > 3) { X(f); }",
            "an rt if condition must read a real-time value; use if",
        ),
        ("rt if (m) { X(f); }", "an rt if condition must be a true/false value"),
        (
            "rt if (m < 4) { qbool z = false; }",
            "'z' cannot be declared inside an rt if body",
        ),
        (
            "if (m < 4) { X(f); }",
            "'m' is a real-time value, which has no value while the circuit is "
            "being built; use rt if",
        ),
    ],
    ids=["quantum", "no real-time value", "not boolean", "declaration", "wrong if"],
)
def test_an_rt_if_is_checked(diagnostics_of: DiagnosticsOf, program: str, message: str) -> None:
    (diagnostic,) = diagnostics_of(_RT_DECLS + program + "\n").errors
    assert diagnostic.message == message


def test_a_measurement_cannot_be_declared_build_time(
    diagnostics_of: DiagnosticsOf,
) -> None:
    (diagnostic,) = diagnostics_of("qbool q = false;\nint r = measure(q);\n").errors
    assert diagnostic.message == ("a measurement result is real-time; declare 'r' as rt int<>")


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "rt int<4> c = 1.5;",
            "a real-time variable holds a whole number or a "
            "truth value, so 'c' cannot start from 1.5",
        ),
        ("rt int<4> c = 0 - 3;", "a real-time variable is unsigned, so 'c' cannot start from -3"),
        ("rt int<2> c = 12;", "12 does not fit in the 2 bits of 'c'"),
        ("rt int<0> c = 0;", "'c' must be at least one bit wide"),
        ("rt int<2> w = measure(a);", "'a' measures into 3 bits, not the 2 declared for 'w'"),
    ],
    ids=["float", "negative", "too wide", "zero width", "measured width"],
)
def test_a_realtime_declaration_is_checked(
    diagnostics_of: DiagnosticsOf, declaration: str, message: str
) -> None:
    (diagnostic,) = diagnostics_of(_RT_DECLS + declaration + "\n").errors
    assert diagnostic.message == message


@pytest.mark.parametrize(
    ("declaration", "width"),
    [
        ("rt int<> c = measure(a);", 3),
        ("rt int<> c = 12;", 4),
        ("rt int<6> c = 12;", 6),
        ("rt bool c = true;", 1),
    ],
    ids=["from a measurement", "from a value", "explicit", "bool"],
)
def test_a_realtime_width_is_settled_by_analysis(
    analyzed_ast: AnalyzedAst, declaration: str, width: int
) -> None:
    """Empty angle brackets take the width from the initializer, the same way
    the value's own size settles it elsewhere."""
    ast = analyzed_ast(_RT_DECLS + declaration + "\n")
    node = ast.statements[-1]
    assert isinstance(node, RealtimeDecl)
    assert node.width == width


def test_a_realtime_variable_that_is_not_measured_warns(
    diagnostics_of: DiagnosticsOf,
) -> None:
    bag = diagnostics_of(_RT_DECLS + "rt int<2> c = 1;\n")
    assert not bag.has_errors
    assert any("Aer 0.17.2 crashes" in w.message for w in bag.warnings)


def test_arithmetic_on_a_realtime_value_warns(diagnostics_of: DiagnosticsOf) -> None:
    bag = diagnostics_of(_RT_DECLS + "rt if (m + 1 > 4) { X(f); }\n")
    assert not bag.has_errors
    assert any("cannot run arithmetic" in w.message for w in bag.warnings)


@pytest.mark.parametrize(
    ("condition", "answer"),
    [("m < 100", "true"), ("m > 100", "false"), ("m >= 0", "true")],
)
def test_a_comparison_outside_the_width_warns(
    diagnostics_of: DiagnosticsOf, condition: str, answer: str
) -> None:
    bag = diagnostics_of(f"{_RT_DECLS}rt if ({condition}) {{ X(f); }}\n")
    assert not bag.has_errors
    assert any(f"always {answer}" in w.message for w in bag.warnings)


_ASSIGN_DECLS = (
    "qint<3> a = [];\nqint<2> b = 0;\nqbool f = false;\nint n = 5;\nbool t = true;\n"
    "rt int<> m = measure(a);\n"
)


@pytest.mark.parametrize(
    ("program", "message"),
    [
        ("a = b;", "'a' is a quantum variable, which cannot be assigned to; use reset() or a gate"),
        ("param int p;\np = 1;", "'p' is a param and is bound, not assigned"),
        ("for(int j in range(2)) { j = 5; }", "'j' is a loop variable and cannot be assigned to"),
        ("a[0] = 1;", "only a variable can be assigned to"),
        (
            "int i = 0;\ni = m;",
            "'i' is settled while the circuit is built, so it cannot be given a real-time value",
        ),
        (
            "process g(int i) { i += 1; }\ng(1);",
            "assigning to the process parameter 'i' is not implemented yet; this "
            "is a limitation of the compiler, not an error in the program",
        ),
    ],
    ids=["quantum", "param", "loop variable", "indexed", "real-time value", "process parameter"],
)
def test_an_assignment_target_is_checked(
    diagnostics_of: DiagnosticsOf, program: str, message: str
) -> None:
    (diagnostic,) = diagnostics_of(_ASSIGN_DECLS + program + "\n").errors
    assert diagnostic.message == message


def test_a_name_read_above_its_declaration_is_rejected(
    diagnostics_of: DiagnosticsOf,
) -> None:
    """Every classical variable is a Python variable in the generated file, so
    the build would find nothing there."""
    messages = [
        diagnostic.message
        for diagnostic in diagnostics_of(f"{_ASSIGN_DECLS}X(a[i]);\nint i = 0;\n").errors
    ]
    assert messages == [
        "a quantum register index must be an integer the compiler can compute"
        " -- 'i': it has no value before its declaration",
        "'i' is read before its declaration",
    ]


@pytest.mark.parametrize(
    ("program", "message"),
    [
        (
            "rt if (m == 1) { i = 1; }",
            "'i' is settled when the circuit is built, so assigning to it "
            "inside an rt if body would happen whatever the measurement says",
        ),
        (
            "rt while (m == 1) { i = 1; reset(a); m = measure(a); }",
            "'i' is settled when the circuit is built, so assigning to it "
            "inside an rt while body would happen once, not on every pass",
        ),
    ],
    ids=["rt if", "rt while"],
)
def test_a_build_time_assignment_inside_a_real_time_body_is_rejected(
    diagnostics_of: DiagnosticsOf, program: str, message: str
) -> None:
    """A real-time body's Python runs when the circuit is built, so the
    assignment does not wait for the measurement."""
    (diagnostic,) = diagnostics_of(f"{_ASSIGN_DECLS}int i = 0;\n" + program + "\n").errors
    assert diagnostic.message == message


@pytest.mark.parametrize(
    ("program", "message"),
    [
        (
            "b *= b;",
            "'b' is a quantum variable, and a product needs a register of its "
            "own to land in; declare one, as in 'qint<n> r = a * b'",
        ),
        (
            "b /= b;",
            "'b' is a quantum variable, and division is not reversible, so it has no circuit",
        ),
    ],
    ids=["times-equals", "divide-equals"],
)
def test_an_operator_a_quantum_variable_cannot_have(
    diagnostics_of: DiagnosticsOf, program: str, message: str
) -> None:
    """`+=` and `-=` describe a reversible adder; these two do not, and
    without the check the lowering would quietly add instead."""
    (diagnostic,) = diagnostics_of(_ASSIGN_DECLS + program + "\n").errors
    assert diagnostic.message == message


def test_a_classical_variable_takes_every_compound_operator(
    diagnostics_of: DiagnosticsOf,
) -> None:
    assert not diagnostics_of(
        f"{_ASSIGN_DECLS}float w = 1.0;\nw *= 2.0;\nw /= 4.0;\nRX(w, f);\n"
    ).has_errors


def test_a_loop_bound_says_why_it_has_no_value(diagnostics_of: DiagnosticsOf) -> None:
    (diagnostic,) = diagnostics_of(
        f"{_ASSIGN_DECLS}int k = 2;\nfor(int j in range(2)) {{ k = 3; }}\n"
        "for(int j2 in range(k)) { X(f); }\n"
    ).errors
    assert diagnostic.message == (
        "a for loop needs an iteration count known when the circuit is built; "
        "'k': its value here depends on a branch"
    )


@pytest.mark.parametrize(
    "program",
    [
        "int i = 0;\ni = 1;\nX(a[i]);",
        "int i = 0;\ni = i + 1;\nX(a[i]);",
        "int i = 0;\ni += 2;\nX(a[i]);",
        "int i = 0;\nif (t) { i = 1; X(a[i]); }",
        "m = measure(a);",
        "m = m ^ 1;",
    ],
)
def test_an_accepted_assignment(diagnostics_of: DiagnosticsOf, program: str) -> None:
    assert not diagnostics_of(_ASSIGN_DECLS + program + "\n").has_errors


_LOOP_DECLS = (
    "qint<3> a = [];\nqint<2> b = 0;\nqbool f = false;\nint n = 5;\nbool t = true;\n"
    "rt int<> m = measure(a);\n"
)


@pytest.mark.parametrize(
    ("program", "message"),
    [
        (
            "rt while (m != 0) { X(f); }",
            "this rt while loop never ends: its body writes nothing its "
            "condition reads, and it has no break",
        ),
        (
            "rt while (b == 2) { X(f); }",
            "'b' is a quantum variable, so it cannot be an rt while condition; use qif",
        ),
        (
            "rt while (n > 3) { X(f); }",
            "an rt while condition must read a real-time value; a loop "
            "settled when the circuit is built is a for",
        ),
        (
            "rt while (m != 0) { qbool z = false; reset(a); m = measure(a); }",
            "'z' cannot be declared inside an rt while body",
        ),
        ("break;", "'break' is only meaningful inside an rt while loop"),
        ("continue;", "'continue' is only meaningful inside an rt while loop"),
    ],
    ids=[
        "rt never settles",
        "quantum condition",
        "build-time condition",
        "declaration in the body",
        "stray break",
        "stray continue",
    ],
)
def test_a_loop_is_checked(diagnostics_of: DiagnosticsOf, program: str, message: str) -> None:
    (diagnostic,) = diagnostics_of(_LOOP_DECLS + program + "\n").errors
    assert diagnostic.message == message


@pytest.mark.parametrize(
    "program",
    [
        "rt while (m != 0) { reset(a); H(a); m = measure(a); }",
        "rt while (m != 0) { reset(a); m = measure(a); rt if (m == 0) { break; } }",
        "rt while (m != 0) { reset(a); m = measure(a); if (t) { break; } }",
    ],
    ids=[
        "repeat until success",
        "break in an rt if",
        "real-time break under a build-time if",
    ],
)
def test_an_accepted_loop(diagnostics_of: DiagnosticsOf, program: str) -> None:
    assert not diagnostics_of(_LOOP_DECLS + program + "\n").has_errors


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "qubit[3] a = 0;",
            "'a' is a sequence of qubits, not a number; leave it at |0> or give "
            "a state in [] or {}",
        ),
        (
            "qubit[1] a = true;",
            "'a' is a sequence of qubits, not a number; leave it at |0> or give "
            "a state in [] or {}",
        ),
        (
            "qubit[2] a;\nqubit[2] b;\na += b;",
            "'a' is a sequence of qubits, not a number, so arithmetic has no meaning on it",
        ),
        (
            "qubit[2] a;\nqint<4> c = 0;\nqint<2> d = 1;\nc += d * a;",
            "'a' is a sequence of qubits, not a number, so arithmetic has no meaning on it",
        ),
        (
            "qubit[2] a;\nqbool t = false;\nqif(a == 2) { X(t); }",
            "'a' is a sequence of qubits, not a number; test one of its qubits instead",
        ),
        (
            "qubit[2] a;\nqbool t = false;\nqif(a) { X(t); }",
            "'a' has 2 qubits; a bare qif condition needs exactly one -- index a "
            "single qubit of it",
        ),
    ],
)
def test_a_qubit_array_is_not_a_number(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """The point of the type: n qubits with no numeric meaning, so everything
    that reads them as one value is rejected."""
    assert [error.message for error in diagnostics_of(source).errors] == [expected]


@pytest.mark.parametrize(
    "source",
    [
        "qubit[2] a;\nqubit[2] b = a;",
        "qint<2> a = 1;\nqint<2> b = a;",
        "qbool a = true;\nqbool b = a;",
    ],
)
def test_a_quantum_variable_cannot_be_copied(diagnostics_of: DiagnosticsOf, source: str) -> None:
    """Section 3.1's no-cloning ban. It used to surface as an unimplemented
    initializer, which blamed the compiler for a rule of physics."""
    assert [error.message for error in diagnostics_of(source).errors] == [
        "'b' cannot copy the quantum variable 'a'; no-cloning forbids it -- measure 'a' instead"
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("qubit[] a;", "'a' has no length to take from its initializer; write one out"),
        ("qint<> a = [];", "'a' has no width to take from its initializer; write one out"),
        ("qubit[] a = {};", "'a' has no length to take from its initializer; write one out"),
        ("qint<> a = [0.5, 0.3, 0.2];", "'a' needs 2^n probabilities, got 3"),
        ("qubit[] a = {1, 0, 0};", "'a' needs 2^n amplitudes, got 3"),
    ],
)
def test_a_size_that_cannot_be_derived_is_reported(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """An empty list is equal superposition over however many qubits there are,
    which is exactly what is missing here."""
    assert [error.message for error in diagnostics_of(source).errors] == [expected]


@pytest.mark.parametrize(
    "source",
    [
        "qubit[3] a;",
        "qubit[3] a;\nH(a);",
        "qubit[2] a;\nCX(a[0], a[1]);",
        "qubit[2] a = [0.5, 0, 0, 0.5];",
        "qubit[1] a = {1 / sqrt(2), 1 / sqrt(2)};",
        "qubit[2] a = [];",
        "qubit[2] a;\nreset(a);",
        "qubit[2] a;\nqbool t = false;\nqif(a[0]) { X(t); }",
        "qubit[2] a;\nrt int<> m = measure(a);",
        "int n = 2;\nqubit[n + 1] a;\nX(a[2]);",
    ],
)
def test_a_qubit_array_is_accepted(diagnostics_of: DiagnosticsOf, source: str) -> None:
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "qint<2> a = 0;\nrt bool[] m = measure(a);\nrt if (m == 2) { X(a[0]); }",
            "'m' is a sequence of bits; index it to use one",
        ),
        (
            "qint<2> a = 0;\nrt bool[] m = measure(a);\nm = 3;",
            "'m' is a sequence of bits; index it to use one",
        ),
        (
            "qint<2> a = 0;\nrt bool[] m = measure(a);\nrt if (m[0] == 1) { X(a[0]); }",
            "a bit of 'm' is already a truth value; test it on its own, or negate it with !",
        ),
        (
            "qint<2> a = 0;\nrt bool[] m = measure(a);\nrt if (m[5]) { X(a[0]); }",
            "index 5 is out of range for 'm' of size 2",
        ),
        (
            "rt bool[2] m = 3;",
            "'m' is the bits of a measurement, so it must start from one",
        ),
        (
            "qint<3> a = 0;\nrt bool[2] m = measure(a);",
            "'a' measures into 3 bits, not the 2 declared for 'm'",
        ),
    ],
)
def test_a_bit_sequence_is_used_one_bit_at_a_time(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    assert diagnostics_of(source).errors[0].message == expected


def test_a_number_cannot_be_indexed_as_bits(diagnostics_of: DiagnosticsOf) -> None:
    """`rt int<n>` is one number of n bits, not n bits -- indexing it used to
    fall through to the condition check, which blamed the wrong thing."""
    bag = diagnostics_of("qint<2> a = 0;\nrt int<2> r = measure(a);\nrt if (r[0]) { X(a[0]); }")
    assert bag.errors[0].message == (
        "'r' is a number, not a sequence of bits, so it cannot be indexed"
    )


@pytest.mark.parametrize(
    "source",
    [
        "qint<2> a = 0;\nrt bool[] m = measure(a);\nrt if (m[0]) { X(a[0]); }",
        "qint<2> a = 0;\nrt bool[] m = measure(a);\nrt if (!m[1]) { X(a[0]); }",
        "qint<2> a = 0;\nrt bool[] m = measure(a);\nrt if (m[0] && m[1]) { X(a[0]); }",
        "qubit[2] a;\nrt bool[2] m = measure(a);\nrt if (m[0]) { X(a[0]); }",
        "qbool q = false;\nrt bool[] m = measure(q);\nrt if (m[0]) { X(q); }",
        "qint<2> a = 0;\nrt bool[] m = measure(a);\nm = measure(a);\nrt if (m[0]) { X(a[0]); }",
        "qint<2> a = 0;\nrt int<> r = measure(a);\nrt bool[] m = measure(a);\n"
        "rt if (m[0] && r == 1) { X(a[0]); }",
    ],
)
def test_a_bit_sequence_is_accepted(diagnostics_of: DiagnosticsOf, source: str) -> None:
    assert not diagnostics_of(source).has_errors


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "process f(qubit[] x) { X(x[0]); }\nqint<3> a = 0;\nf(a);",
            "'f' takes a qubit sequence for 'x', but got a qint",
        ),
        (
            "process f(qint x) { X(x); }\nqbool q = false;\nf(q);",
            "'f' takes a qint for 'x', but got a qbool",
        ),
        (
            "process f(qbool x) { X(x); }\nqubit[1] a;\nf(a);",
            "'f' takes a qbool for 'x', but got a qubit sequence",
        ),
        (
            "process f(int x) { }\nqint<2> a = 0;\nf(a);",
            "'f' takes a classical value for 'x', but got a quantum variable",
        ),
        (
            "process f(int x) { }\nqint<2> a = 0;\nf(a[0]);",
            "'f' takes a classical value for 'x', but got a quantum variable",
        ),
    ],
)
def test_a_parameter_type_says_what_may_arrive(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """The declared type used to be documentation: a qbool went into a qint
    parameter and a quantum variable into a classical one, and the only
    complaint came from the inlined body, pointing inside the process."""
    assert [error.message for error in diagnostics_of(source).errors] == [expected]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "process f(qint<4> x) { X(x[0]); }\nqint<2> a = 0;\nf(a);",
            "'f' takes 4 qubit(s) for 'x', but got 2",
        ),
        (
            "process f(qubit[4] x) { X(x[0]); }\nqubit[2] a;\nf(a);",
            "'f' takes 4 qubit(s) for 'x', but got 2",
        ),
        (
            "int n = 3;\nprocess f(qint<n> x) { X(x[0]); }\nqint<2> a = 0;\nf(a);",
            "'f' takes 3 qubit(s) for 'x', but got 2",
        ),
        (
            "process f(qint<4> x) { X(x); }\nqint<4> a = 0;\nf(a[0]);",
            "'f' takes 4 qubit(s) for 'x', but got 1",
        ),
    ],
)
def test_a_written_parameter_width_binds(
    diagnostics_of: DiagnosticsOf, source: str, expected: str
) -> None:
    """Checked after expansion, because a process body may change a classical
    variable a width reads -- so no width is settled while the call still is."""
    assert [error.message for error in diagnostics_of(source).errors] == [expected]


@pytest.mark.parametrize(
    "source",
    [
        "process f(qubit[] x) { X(x[0]); }\nqubit[3] a;\nf(a);",
        "process f(qubit[3] x) { H(x); }\nqubit[3] a;\nf(a);",
        "process f(qint<4> x) { X(x[0]); }\nqint<4> a = 0;\nf(a);",
        "process f(qint x) { X(x); }\nqint<3> a = 0;\nf(a[0]);",
        "process f(qbool x) { X(x); }\nqbool q = false;\nf(q);",
        "process f(qint x, qubit[] y) { X(x); H(y); }\nqint<2> a = 0;\nqubit[2] b;\nf(a, b);",
        "process f(float t) { }\nparam float th;\nf(th);",
        "process f(int k) { }\nint n = 1;\nf(n);",
    ],
)
def test_a_matching_argument_is_accepted(diagnostics_of: DiagnosticsOf, source: str) -> None:
    assert not diagnostics_of(source).has_errors


def test_a_parameter_width_cannot_name_a_parameter(diagnostics_of: DiagnosticsOf) -> None:
    """A width says how wide the process is, so it reads the scope around the
    process -- not the parameters it is being written beside."""
    bag = diagnostics_of("process f(qint<x> x) { X(x[0]); }\nqint<2> a = 0;\nf(a);")
    assert bag.errors[0].message == "undefined name 'x'"
