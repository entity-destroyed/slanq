"""The process-expansion pass. Run through `analyze`, since expansion sits
between name resolution and type checking and depends on the first."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from slanq.analysis import analyze
from slanq.ast_nodes import (
    AugAssign,
    BinaryOp,
    ExprStatement,
    Index,
    Name,
    ProcessDef,
    Program,
    QIf,
)
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
    """`int r = measure(x)` is a declaration, so it hits the same limit -- and
    it has the same failure mode, a second `r` register per extra call."""
    _, bag = expanded(
        "process f(qint x) { int r = measure(x); }\nqint<2> a = 0;\nf(a);\n"
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
    ast, bag = expanded("process f(qint x) { if(1) { X(x); } }\nqint<2> a = 0;\nf(a);\n")
    assert not bag.has_errors  # analysis is fine; lowering is what refuses it
    lower_to_ir(ast, bag)
    assert any("if statement" in message for message in _messages(bag))
