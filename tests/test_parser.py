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


def test_mvp_c_parses(mvp_c_source: str) -> None:
    assert parse_source(mvp_c_source) is not None

@pytest.mark.parametrize(
    ("source", "line", "column", "found"),
    [
        ("qbool q = false;;\n", 1, 17, "';'"),
        ("qint<-1> a = 0;\n", 1, 6, "'-'"),
        ("qbool q = false;\nqif(1 < 2 < 3) { X(q); }\n", 2, 11, "'<'"),
    ],
)
def test_a_syntax_error_names_what_it_found(
    source: str, line: int, column: int, found: str
) -> None:
    with pytest.raises(SlanqError) as error:
        parse_source(source)
    assert f"line {line}, column {column}" in str(error.value)
    assert f"unexpected {found}" in str(error.value)


def test_running_out_of_input_points_at_the_end_of_the_text() -> None:
    with pytest.raises(SlanqError) as error:
        parse_source("qbool q = false\n")
    assert "line 1, column 16: unexpected end of file" in str(error.value)


@pytest.mark.parametrize(
    ("source", "bracket", "line", "column"),
    [
        ("qint<2> a = 2;\nqif(a == 2) { X(a[0]);\n", "'{'", 2, 13),
        ("qbool q = false;\nRX(1.5, q\n", "'('", 2, 3),
        ("qint<2> a = 2;\nX(a[0\n", "'['", 2, 4),
    ],
)
def test_an_unclosed_bracket_says_where_it_was_opened(
    source: str, bracket: str, line: int, column: int
) -> None:
    with pytest.raises(SlanqError) as error:
        parse_source(source)
    assert f"{bracket} at line {line}, column {column} is never closed" in str(
        error.value
    )


def test_a_bracket_inside_a_comment_is_not_counted() -> None:
    with pytest.raises(SlanqError) as error:
        parse_source("// {\nqbool q = false\n")
    assert "is never closed" not in str(error.value)
    assert "unexpected end of file" in str(error.value)


def test_a_mismatched_bracket_does_not_claim_an_unclosed_one() -> None:
    with pytest.raises(SlanqError) as error:
        parse_source("qint<2> a = 2;\nX(a[0);\n")
    assert "is never closed" not in str(error.value)


@pytest.mark.parametrize("source", ["", "// nothing\n", "\n\n   \n"])
def test_an_empty_program_parses(source: str) -> None:
    assert parse_source(source) is not None


@pytest.mark.parametrize(
    "source",
    ["qbool ψ = false;", "qint<3> φ = 0;", "int θ = 1;"],
)
def test_a_name_may_use_letters_beyond_ascii(source: str) -> None:
    assert parse_source(source) is not None
