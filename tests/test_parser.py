from __future__ import annotations

import pytest

from slanq.diagnostics import SlanqError
from slanq.parser import parse_source


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
        "for(int i in range(4)) { num += gamma[i]; }",
        "while(result == 0) { result = measure(num); }",
        "process add(qint a, qint b) { a += b; }",
        "// just a comment",
    ],
)
def test_parses_language_constructs(source: str) -> None:
    assert parse_source(source) is not None


def test_example_program_parses(hello_source: str) -> None:
    assert parse_source(hello_source) is not None


def test_mvp_a_parses(mvp_a_source: str) -> None:
    assert parse_source(mvp_a_source) is not None


def test_syntax_error_is_reported() -> None:
    with pytest.raises(SlanqError):
        parse_source("qint<3> a = ;")


@pytest.mark.parametrize(
    "source",
    [
        "float r = 7 / 2;",
        "int r = 7 % 3;",
        "int r = 2 ** 8;",
        "int r = -3;",
        "float r = -PI / 2;",
        "int r = floor(7 / 2);",
        "for(int i in range(1, 8, 2)) { X(q[0]); }",
    ],
)
def test_parses_arithmetic_and_loops(source: str) -> None:
    assert parse_source(source) is not None


@pytest.mark.parametrize("source", ["int r = 1 < 2 < 3;", "int r = 1 == 2 == 3;"])
def test_chained_comparison_is_rejected(source: str) -> None:
    """Python would chain it and C would nest it; the two disagree, so neither."""
    with pytest.raises(SlanqError):
        parse_source(source)


def test_double_slash_is_a_comment_not_integer_division() -> None:
    with pytest.raises(SlanqError):
        parse_source("int r = 7 // 2;")
