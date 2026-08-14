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
        "for(int i = 0; i < 4; i++) { num += gamma[i]; }",
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
