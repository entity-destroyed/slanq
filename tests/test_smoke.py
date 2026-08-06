"""Basic smoke tests: the package imports, the grammar is reachable and parses."""

from __future__ import annotations

from pathlib import Path

import pytest

import slanq
from slanq.diagnostics import DiagnosticBag, Severity, SlanqError
from slanq.grammar import grammar_text
from slanq.parser import parse_source

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_version_is_exposed() -> None:
    assert slanq.__version__


def test_grammar_is_packaged() -> None:
    """The .lark file must be part of the package, not just of the source tree.

    This test fails if the package data declaration is missing from
    pyproject.toml.
    """
    text = grammar_text()
    assert "start: statement*" in text


@pytest.mark.parametrize(
    "source",
    [
        "qint<3> a = 0;",
        "qint<2> b = [];",
        "qint<2> c = [0, 0.5, 0.5, 0];",
        "qbool flag = true;",
        "param float theta;",
        "param int gamma[4];",
        "int r = measure(a);",
        "a += b;",
        "X(a[1]);",
        "CX(b[2], a[0]);",
        "qif(a == 2) { phase(60); }",
        "if(x == y) { a += 1; }",
        "for(int i = 0; i < 4; i++) { num += gamma[i]; }",
        "while(result == 0) { result = measure(num); }",
        "process add(qint a, qint b) { a += b; }",
        "// just a comment",
    ],
)
def test_parses_language_constructs(source: str) -> None:
    assert parse_source(source) is not None


def test_example_program_parses() -> None:
    source = (EXAMPLES / "hello.slanq").read_text(encoding="utf-8")
    assert parse_source(source) is not None


def test_syntax_error_is_reported() -> None:
    with pytest.raises(SlanqError):
        parse_source("qint<3> a = ;")


def test_diagnostic_bag_separates_errors_and_warnings() -> None:
    bag = DiagnosticBag()
    bag.warning("too many qubits", line=3)
    assert not bag.has_errors

    bag.error("no-cloning violation", line=5, column=9)
    assert bag.has_errors
    assert len(bag.errors) == 1
    assert len(bag.warnings) == 1
    assert bag.errors[0].severity is Severity.ERROR

    with pytest.raises(SlanqError):
        bag.raise_if_errors()
