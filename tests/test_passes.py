"""The process-expansion and loop-unrolling passes. Run through `analyze`,
since both sit between name resolution and type checking and depend on the
first."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import (
    AugAssign,
    BinaryOp,
    Call,
    ExprStatement,
    For,
    If,
    Index,
    Literal,
    Name,
    ProcessDef,
    Program,
    QIf,
    Statement,
    UnknownValue,
)
from slanq.builtin import const_value
from slanq.diagnostics import DiagnosticBag
from slanq.lowering import lower_to_ir
from slanq.passes import MAX_EXPANDED_STATEMENTS

BuildAst = Callable[[str], Program]
Expanded = Callable[[str], tuple[Program, DiagnosticBag]]


@pytest.fixture
def expanded(build_ast: BuildAst) -> Expanded:
    def _expanded(source: str) -> tuple[Program, DiagnosticBag]:
        ast = build_ast(source)
        bag = DiagnosticBag()
        analyze(ast, bag)
        return ast, bag

    return _expanded


def _messages(bag: DiagnosticBag) -> list[str]:
    return [diagnostic.message for diagnostic in bag.errors]


def test_call_becomes_the_body_with_arguments_substituted(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process add(qint x, qint y) { x += y; }\n"
        "qint<2> num1 = 1;\nqint<2> num2 = 2;\nadd(num1, num2);\n"
    )
    assert not bag.has_errors
    assignment = ast.statements[-1]
    assert isinstance(assignment, AugAssign)
    assert isinstance(assignment.target, Name) and assignment.target.name == "num1"
    assert isinstance(assignment.value, Name) and assignment.value.name == "num2"


def test_substituted_name_keeps_its_resolved_symbol(expanded: Expanded) -> None:
    """The whole point of expanding after name resolution: the copy is already
    resolved, so no second resolution pass is needed."""
    ast, bag = expanded(
        "process flip(qbool b) { X(b); }\nqbool q = false;\nflip(q);\n"
    )
    assert not bag.has_errors
    statement = ast.statements[-1]
    assert isinstance(statement, ExprStatement)
    (argument,) = statement.expr.args  # type: ignore[attr-defined]
    assert isinstance(argument, Name)
    assert argument.resolved_symbol is ast.statements[1]


def test_the_definition_stays_in_the_tree(expanded: Expanded) -> None:
    """It is a template, not code: kept so its body is still type-checked,
    and lowered to nothing."""
    ast, _ = expanded("process f(qint x) { X(x); }\nqint<2> a = 0;\nf(a);\n")
    assert isinstance(ast.statements[0], ProcessDef)


def test_an_indexed_argument_is_substituted_verbatim(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process flip(qbool b) { X(b); }\nqint<2> a = 0;\nflip(a[1]);\n"
    )
    assert not bag.has_errors
    statement = ast.statements[-1]
    assert isinstance(statement, ExprStatement)
    (argument,) = statement.expr.args  # type: ignore[attr-defined]
    assert isinstance(argument, Index)


def test_a_classical_argument_is_substituted_as_an_expression(expanded: Expanded) -> None:
    """A classical parameter needs no runtime representation: the caller's
    expression takes its place, so const_value sees it unchanged."""
    ast, bag = expanded(
        "process rot(qbool q, float angle) { RX(angle, q); }\n"
        "qbool t = false;\nrot(t, PI / 2);\n"
    )
    assert not bag.has_errors
    statement = ast.statements[-1]
    assert isinstance(statement, ExprStatement)
    angle = statement.expr.args[0]  # type: ignore[attr-defined]
    # Unfolded, as everywhere else: the expression is substituted, not its value.
    assert isinstance(angle, BinaryOp) and angle.op == "/"


def test_two_calls_expand_independently(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process bump(qint x) { x += 1; }\nqint<2> a = 0;\nbump(a);\nbump(a);\n"
    )
    assert not bag.has_errors
    first, second = ast.statements[-2], ast.statements[-1]
    assert isinstance(first, AugAssign) and isinstance(second, AugAssign)
    assert first is not second


def test_a_nested_call_is_expanded_too(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process inner(qbool x) { X(x); }\nprocess outer(qbool y) { inner(y); }\n"
        "qbool q = false;\nouter(q);\n"
    )
    assert not bag.has_errors
    statement = ast.statements[-1]
    assert isinstance(statement, ExprStatement)
    assert statement.expr.callee.name == "X"  # type: ignore[attr-defined]


def test_a_qif_in_the_body_survives_expansion(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process guarded(qint a, qbool out) { qif(a == 2) { X(out); } }\n"
        "qint<2> a = 2;\nqbool out = false;\nguarded(a, out);\n"
    )
    assert not bag.has_errors
    assert isinstance(ast.statements[-1], QIf)


def test_a_call_may_precede_its_definition(expanded: Expanded) -> None:
    _, bag = expanded(
        "qbool q = false;\nflip(q);\nprocess flip(qbool b) { X(b); }\n"
    )
    assert not bag.has_errors


def test_a_parameter_may_shadow_a_top_level_name(expanded: Expanded) -> None:
    """A body's `x` is the parameter, never the global of the same name, and
    substitution replaces it with the argument -- which here happens to be
    that same global, but only because the call says so."""
    ast, bag = expanded("qint<2> x = 0;\nprocess f(qint x) { x += 1; }\nf(x);\n")
    assert not bag.has_errors
    assignment = ast.statements[-1]
    assert isinstance(assignment, AugAssign)
    assert isinstance(assignment.target, Name)
    assert assignment.target.resolved_symbol is ast.statements[0]


def test_a_parameter_used_twice_gets_two_copies(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process twice(qbool b) { X(b); X(b); }\nqbool q = true;\ntwice(q);\n"
    )
    assert not bag.has_errors
    first, second = ast.statements[-2], ast.statements[-1]
    assert isinstance(first, ExprStatement) and isinstance(second, ExprStatement)
    (left,) = first.expr.args  # type: ignore[attr-defined]
    (right,) = second.expr.args  # type: ignore[attr-defined]
    assert left is not right
    assert left.resolved_symbol is right.resolved_symbol


def test_a_classical_parameter_can_be_a_qubit_index(expanded: Expanded) -> None:
    """The index is a compile-time value only after substitution, which is
    exactly why expansion runs before the index checks."""
    _, bag = expanded(
        "process poke(qint q, int i) { X(q[i]); }\nqint<2> a = 0;\npoke(a, 1);\n"
    )
    assert not bag.has_errors


def test_an_index_from_an_argument_is_bounds_checked(expanded: Expanded) -> None:
    _, bag = expanded(
        "process poke(qint q, int i) { X(q[i]); }\nqint<2> a = 0;\npoke(a, 5);\n"
    )
    assert any("index 5 is out of range" in message for message in _messages(bag))


def test_self_recursion_is_rejected(expanded: Expanded) -> None:
    _, bag = expanded("process f(qbool x) { f(x); }\nqbool a = false;\nf(a);\n")
    assert any("calls itself" in message for message in _messages(bag))


def test_mutual_recursion_is_rejected(expanded: Expanded) -> None:
    """The active-process stack catches a cycle of any length, not just a
    direct self-call."""
    _, bag = expanded(
        "process f(qbool x) { g(x); }\nprocess g(qbool x) { f(x); }\n"
        "qbool a = false;\nf(a);\n"
    )
    assert any("calls itself" in message for message in _messages(bag))


def test_a_nested_definition_is_rejected(expanded: Expanded) -> None:
    _, bag = expanded("qbool q = false;\nqif(q) { process f(qbool x) { X(x); } }\n")
    assert any("only be defined at the top level" in m for m in _messages(bag))


def test_a_body_declaration_is_reported_as_a_limitation(expanded: Expanded) -> None:
    """Copied once per call, two copies would claim the same register name and
    the generated file would fail to build."""
    _, bag = expanded(
        "process f(qint x) { qint<2> t = 0; x += t; }\nqint<2> a = 0;\nf(a);\n"
    )
    messages = _messages(bag)
    assert messages == [
        "a declaration inside a process body is not implemented yet; this is a "
        "limitation of the compiler, not an error in the program"
    ]


def test_a_body_declaration_is_reported_once_for_two_calls(expanded: Expanded) -> None:
    _, bag = expanded(
        "process f(qint x) { qint<2> t = 0; x += t; }\nqint<2> a = 0;\nf(a);\nf(a);\n"
    )
    assert len(_messages(bag)) == 1


def test_a_body_declaration_is_reported_even_if_never_called(expanded: Expanded) -> None:
    _, bag = expanded("process f(qint x) { qint<2> t = 0; x += t; }\n")
    assert any("declaration inside a process body" in m for m in _messages(bag))


def test_a_measure_in_a_body_is_a_declaration_too(expanded: Expanded) -> None:
    """`rt int<> r = measure(x)` is a declaration, so it hits the same limit -- and
    it has the same failure mode, a second `r` register per extra call."""
    _, bag = expanded(
        "process f(qint x) { rt int<> r = measure(x); }\nqint<2> a = 0;\nf(a);\n"
    )
    assert any("declaration inside a process body" in m for m in _messages(bag))


def test_indexing_a_parameter_needs_a_whole_variable(expanded: Expanded) -> None:
    """Substituting `a[1]` for an indexed parameter would build `a[1][0]`,
    which no later phase can read."""
    _, bag = expanded("process f(qint x) { X(x[0]); }\nqint<2> a = 0;\nf(a[1]);\n")
    assert _messages(bag) == [
        "a process parameter that is indexed in the process body must be given "
        "a whole quantum variable, not a single qubit"
    ]


def test_an_unbounded_expansion_is_rejected_without_hanging(expanded: Expanded) -> None:
    """Not recursive, so the cycle check does not fire: each level calls the
    next one twice, which is 2^20 statements by the bottom."""
    source = "".join(
        f"process p{i}(qbool x) {{ p{i + 1}(x); p{i + 1}(x); }}\n" for i in range(20)
    )
    source += "process p20(qbool x) { X(x); }\nqbool a = false;\np0(a);\n"
    _, bag = expanded(source)
    assert any(str(MAX_EXPANDED_STATEMENTS) in m for m in _messages(bag))


def test_an_unimplemented_statement_in_a_body_still_reports_itself(
    expanded: Expanded,
) -> None:
    """Nothing needs to gate these: the copy reaches lowering like any other
    statement and is reported there."""
    ast, bag = expanded(
        "process f(qbool x) { qif(a == 2) { reset(x); } }\n"
        "qint<2> a = 2;\nqbool q = false;\nf(q);\n"
    )
    assert not bag.has_errors  # analysis is fine; lowering is what refuses it
    lower_to_ir(ast, bag)
    assert any("unitary operations" in message for message in _messages(bag))


def test_a_body_error_is_reported_once_for_two_call_sites(expanded: Expanded) -> None:
    """Every copy of a body shares the span of the code as written, so the two
    call sites and the definition itself all report the same mistake at the
    same place -- three times, before the bag started deduplicating."""
    _, bag = expanded(
        "qint<2> a = 0;\nqint<2> b = 0;\n"
        "process f(qint x) {\n    x += 1.5;\n}\nf(a);\nf(b);\n"
    )
    assert len(_messages(bag)) == 1


def _warnings(bag: DiagnosticBag) -> list[str]:
    return [diagnostic.message for diagnostic in bag.warnings]


def _gate_indices(statements: list[Statement]) -> list[object]:
    """The index each gate call in `statements` was given, in order."""
    indices: list[object] = []
    for statement in statements:
        assert isinstance(statement, ExprStatement)
        call = statement.expr
        assert isinstance(call, Call)
        argument = call.args[0]
        assert isinstance(argument, Index)
        assert isinstance(argument.index, Literal)
        indices.append(argument.index.value)
    return indices


def test_a_loop_becomes_its_body_once_per_iteration(expanded: Expanded) -> None:
    ast, bag = expanded("qint<3> a = 0;\nfor(int i in range(3)) { X(a[i]); }\n")
    assert not bag.has_errors
    assert _gate_indices(ast.statements[1:]) == [0, 1, 2]


def test_no_loop_survives_unrolling(expanded: Expanded) -> None:
    """Later phases never see a For, which is why lowering treats one as a
    compiler bug rather than an unimplemented statement."""
    ast, _ = expanded("qint<3> a = 0;\nfor(int i in range(3)) { X(a[i]); }\n")
    assert not any(isinstance(statement, For) for statement in ast.statements)


def test_a_start_and_a_step_are_honoured(expanded: Expanded) -> None:
    ast, bag = expanded("qint<8> a = 0;\nfor(int i in range(2, 8, 2)) { X(a[i]); }\n")
    assert not bag.has_errors
    assert _gate_indices(ast.statements[1:]) == [2, 4, 6]


def test_a_negative_step_counts_down(expanded: Expanded) -> None:
    ast, bag = expanded("qint<5> a = 0;\nfor(int i in range(4, 0, -1)) { X(a[i]); }\n")
    assert not bag.has_errors
    assert _gate_indices(ast.statements[1:]) == [4, 3, 2, 1]


def test_the_loop_value_is_substituted_unfolded(expanded: Expanded) -> None:
    """The same rule process parameters follow: the value replaces the
    variable, it does not collapse the expression around it."""
    ast, bag = expanded("qbool q = false;\nfor(int i in range(2)) { RZ(i * PI, q); }\n")
    assert not bag.has_errors
    statement = ast.statements[2]
    assert isinstance(statement, ExprStatement)
    call = statement.expr
    assert isinstance(call, Call)
    angle = call.args[0]
    assert isinstance(angle, BinaryOp)
    assert isinstance(angle.left, Literal)
    assert angle.left.value == 1


def test_an_index_from_the_loop_variable_is_bounds_checked(expanded: Expanded) -> None:
    """The iteration that steps off the end is the one reported, which is only
    visible because the loop was unrolled first."""
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(3)) { X(a[i]); }\n")
    assert any("index 2 is out of range" in message for message in _messages(bag))


def test_a_nested_loop_may_count_from_the_outer_variable(expanded: Expanded) -> None:
    """`range(i)` is not a constant where it is written, only once the loop
    around it has been unrolled -- so the iteration count cannot be judged in
    analysis."""
    ast, bag = expanded(
        "qint<3> a = 0;\nfor(int i in range(3)) { for(int j in range(i)) { X(a[j]); } }\n"
    )
    assert not bag.has_errors
    assert _gate_indices(ast.statements[1:]) == [0, 0, 1]


def test_a_loop_inside_a_process_body_is_unrolled(expanded: Expanded) -> None:
    ast, bag = expanded(
        "process f(qint x) { for(int i in range(3)) { X(x[i]); } }\n"
        "qint<3> a = 0;\nf(a);\n"
    )
    assert not bag.has_errors
    assert _gate_indices(ast.statements[2:]) == [0, 1, 2]


def test_a_process_call_inside_a_loop_is_expanded_per_iteration(
    expanded: Expanded,
) -> None:
    ast, bag = expanded(
        "process bump(qint x) { x += 1; }\nqint<2> a = 0;\n"
        "for(int i in range(3)) { bump(a); }\n"
    )
    assert not bag.has_errors
    assert sum(isinstance(s, AugAssign) for s in ast.statements) == 3


def test_a_loop_in_a_qif_body_is_unrolled_in_place(expanded: Expanded) -> None:
    ast, bag = expanded(
        "qint<2> c = 3;\nqint<2> t = 0;\n"
        "qif(c == 3) { for(int i in range(2)) { X(t[i]); } }\n"
    )
    assert not bag.has_errors
    qif = ast.statements[2]
    assert isinstance(qif, QIf)
    assert _gate_indices(qif.body.statements) == [0, 1]


def test_a_loop_whose_body_touches_a_condition_qubit_is_rejected(
    expanded: Expanded,
) -> None:
    """The qif overlap rule applies to what the loop expands to, not to what it
    is written as -- another check that comes for free after unrolling."""
    _, bag = expanded(
        "qint<2> a = 2;\nqif(a[0]) { for(int i in range(2)) { X(a[i]); } }\n"
    )
    assert any("its own condition tests" in message for message in _messages(bag))


def test_a_runtime_parameter_cannot_set_the_iteration_count(expanded: Expanded) -> None:
    _, bag = expanded(
        "param int n;\nqint<2> a = 0;\nfor(int i in range(n)) { X(a[0]); }\n"
    )
    (message,) = _messages(bag)
    assert "known when the circuit is built" in message
    assert "'n' only gets its value at runtime" in message
    # Not phrased as a limitation: Qiskit's own for_loop cannot do it either.
    assert "not implemented" not in message


def test_a_fractional_iteration_count_is_rejected(expanded: Expanded) -> None:
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(2.0)) { X(a[0]); }\n")
    assert any("whole numbers" in message for message in _messages(bag))


def test_a_zero_step_is_rejected(expanded: Expanded) -> None:
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(0, 4, 0)) { X(a[0]); }\n")
    assert any("must not be zero" in message for message in _messages(bag))


def test_a_failed_evaluation_in_the_range_is_reported(expanded: Expanded) -> None:
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(4 / 0)) { X(a[0]); }\n")
    assert any("division by zero" in message for message in _messages(bag))


def test_a_loop_that_runs_zero_times_is_warned_about(expanded: Expanded) -> None:
    ast, bag = expanded("qint<2> a = 0;\nfor(int i in range(0)) { X(a[0]); }\n")
    assert not bag.has_errors
    assert any("runs zero times" in message for message in _warnings(bag))
    assert len(ast.statements) == 1


def test_an_empty_loop_body_is_warned_about(expanded: Expanded) -> None:
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(3)) { }\nX(a[0]);\n")
    assert not bag.has_errors
    assert any("empty body" in message for message in _warnings(bag))


def test_a_loop_that_runs_zero_times_with_an_empty_body_is_warned_about_once(
    expanded: Expanded,
) -> None:
    """Both are true, and the iteration count is the more useful one to hear."""
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(0)) { }\nX(a[0]);\n")
    assert _warnings(bag) == [
        "this loop runs zero times, so nothing in its body is compiled"
    ]


def test_an_empty_loop_body_in_a_template_is_not_warned_about(
    expanded: Expanded,
) -> None:
    """Same rule as the zero-iteration warning: a process body is a template
    until it is called, and the call sites are where anything is said."""
    _, bag = expanded("process f(int n) { for(int i in range(n)) { } }\n")
    assert not _warnings(bag)


def test_an_empty_pass_of_a_nested_loop_is_not_warned_about(expanded: Expanded) -> None:
    """A triangular loop's first pass is legitimately empty; warning about it
    would fire on ordinary code."""
    _, bag = expanded(
        "qint<3> a = 0;\nfor(int i in range(3)) { for(int j in range(i)) { X(a[j]); } }\n"
    )
    assert _warnings(bag) == []


def test_a_declaration_in_a_loop_body_is_reported_as_a_limitation(
    expanded: Expanded,
) -> None:
    """Two iterations would claim one register name, exactly as two calls of a
    process would."""
    _, bag = expanded(
        "qint<2> a = 0;\nfor(int i in range(2)) { qint<2> t = 0; a += t; }\n"
    )
    (message,) = _messages(bag)
    assert "a declaration inside a loop body is not implemented yet" in message


def test_a_measure_in_a_loop_body_is_a_declaration_too(expanded: Expanded) -> None:
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(2)) { rt int<> r = measure(a); }\n")
    assert any("inside a loop body" in message for message in _messages(bag))


def test_indexing_the_loop_variable_is_rejected(expanded: Expanded) -> None:
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(2)) { X(i[0]); }\n")
    assert any("cannot be indexed" in message for message in _messages(bag))


def test_an_unbounded_loop_is_rejected_without_hanging(expanded: Expanded) -> None:
    """The cost is charged before anything is copied, so a count large enough
    to exhaust memory is never reached one statement at a time."""
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(1000000000)) { X(a[0]); }\n")
    assert any(str(MAX_EXPANDED_STATEMENTS) in message for message in _messages(bag))


def test_an_empty_body_still_costs_an_iteration(expanded: Expanded) -> None:
    """Otherwise the budget never moves and the loop spins for free."""
    _, bag = expanded("qint<2> a = 0;\nfor(int i in range(1000000000)) { }\n")
    assert any(str(MAX_EXPANDED_STATEMENTS) in message for message in _messages(bag))


def test_nested_loops_share_one_budget(expanded: Expanded) -> None:
    _, bag = expanded(
        "qint<2> a = 0;\n"
        "for(int i in range(200)) { for(int j in range(200)) { X(a[0]); } }\n"
    )
    assert any(str(MAX_EXPANDED_STATEMENTS) in message for message in _messages(bag))


def test_a_classical_parameter_can_set_a_loop_count(expanded: Expanded) -> None:
    """The definition stays in the tree as a template where `n` has no value,
    so unrolling it there is impossible however constant the call makes it.
    Reporting from inside a template would reject this valid program."""
    ast, bag = expanded(
        "process f(int n, qint x) { for(int i in range(n)) { X(x[i]); } }\n"
        "qint<3> a = 0;\nf(3, a);\n"
    )
    assert not bag.has_errors
    assert _gate_indices(ast.statements[2:]) == [0, 1, 2]


def test_a_template_loop_that_cannot_be_unrolled_is_dropped_silently(
    expanded: Expanded,
) -> None:
    _, bag = expanded(
        "process f(int n, qint x) { for(int i in range(n)) { X(x[i]); } }\n"
        "qint<3> a = 0;\n"
    )
    assert not bag.has_errors
    assert not bag.warnings


def test_a_loop_in_an_uncalled_process_body_is_still_bounds_checked(
    expanded: Expanded,
) -> None:
    """Silence in a template covers only what the template cannot know. A
    constant range still unrolls there, so this error survives -- which is why
    the pass descends into a definition at all."""
    _, bag = expanded(
        "qint<2> a = 0;\nprocess f() { for(int i in range(3)) { X(a[i]); } }\n"
    )
    assert any("index 2 is out of range" in message for message in _messages(bag))


def test_a_declaration_in_a_template_loop_is_reported_only_when_called(
    expanded: Expanded,
) -> None:
    """The accepted cost of template silence: a call site re-reports this at
    the same span, so nothing is lost unless the process is never called."""
    body = "process f(qint x) { for(int i in range(2)) { qint<2> t = 0; x += t; } }\n"
    _, called = expanded("qint<2> a = 0;\n" + body + "f(a);\n")
    assert any("inside a loop body" in message for message in _messages(called))
    _, uncalled = expanded("qint<2> a = 0;\n" + body)
    assert not uncalled.has_errors


def test_a_silenced_budget_message_does_not_count_as_reported(
    expanded: Expanded,
) -> None:
    """A template overflow consumes the budget but says nothing, so the
    report-once flag must stay clear for the next real overflow."""
    _, bag = expanded(
        "qint<2> a = 0;\n"
        "process f() { for(int i in range(1000000000)) { X(a[0]); } }\n"
        "for(int k in range(1000000000)) { X(a[0]); }\n"
    )
    assert any(str(MAX_EXPANDED_STATEMENTS) in message for message in _messages(bag))


def _index_of(ast: Program, statement: int) -> object:
    """The expression in effect for the name used as a qubit index."""
    expression = ast.statements[statement]
    assert isinstance(expression, ExprStatement)
    (argument,) = expression.expr.args  # type: ignore[attr-defined]
    return const_value(argument.index)


def test_a_name_means_what_was_last_written_to_it(expanded: Expanded) -> None:
    ast, bag = expanded("qint<3> a = 0;\nint i = 0;\nX(a[i]);\ni = i + 1;\nX(a[i]);\n")
    assert not bag.has_errors
    assert _index_of(ast, 2) == 0
    assert _index_of(ast, 4) == 1


def test_a_compound_assignment_folds_into_the_value(expanded: Expanded) -> None:
    ast, bag = expanded("qint<3> a = 0;\nint i = 0;\ni += 2;\nX(a[i]);\n")
    assert not bag.has_errors
    assert _index_of(ast, 3) == 2


def test_a_value_written_in_a_branch_is_unknown_after_it(expanded: Expanded) -> None:
    """Two arms meet and the compiler cannot name one value, so it says so
    rather than picking the one it happened to walk through."""
    ast, _ = expanded("qint<3> a = 0;\nint i = 0;\nbool t = true;\nif (t) { i = 1; }\nX(a[i]);\n")
    statement = ast.statements[4]
    assert isinstance(statement, ExprStatement)
    (argument,) = statement.expr.args  # type: ignore[attr-defined]
    assert isinstance(argument.index.effective_value, UnknownValue)
    assert "branch" in argument.index.effective_value.reason


def test_a_value_inside_a_branch_is_still_known(expanded: Expanded) -> None:
    ast, bag = expanded("qint<3> a = 0;\nint i = 0;\nbool t = true;\nif (t) { i = 1; X(a[i]); }\n")
    assert not bag.has_errors
    branch = ast.statements[3]
    assert isinstance(branch, If)
    assert _index_of(branch.body, 1) == 1  # type: ignore[arg-type]


def test_a_name_used_before_its_declaration_has_no_value(expanded: Expanded) -> None:
    """Every classical variable is a Python variable in the generated file, so
    reading one above its own assignment has nothing to read."""
    ast, _ = expanded("qint<3> a = 0;\nX(a[i]);\nint i = 1;\n")
    statement = ast.statements[1]
    assert isinstance(statement, ExprStatement)
    (argument,) = statement.expr.args  # type: ignore[attr-defined]
    assert isinstance(argument.index.effective_value, UnknownValue)
