from __future__ import annotations

import pytest

from slanq.compiler import compile_source
from slanq.diagnostics import SlanqError


def test_full_pipeline_produces_qiskit_source(mvp_a_source: str) -> None:
    result = compile_source(mvp_a_source, source_name="mvp_a.slanq")
    assert not result.diagnostics.has_errors
    assert result.qiskit_source is not None
    assert "def build_circuit()" in result.qiskit_source


@pytest.mark.parametrize(
    ("stage", "reached", "not_reached"),
    [
        ("parse", "parse_tree", "ast"),
        ("ast", "ast", "ir"),
        ("semantic", "ast", "ir"),
        ("ir", "ir", "qiskit_source"),
    ],
)
def test_stop_after_halts_the_pipeline(
    mvp_a_source: str, stage: str, reached: str, not_reached: str
) -> None:
    result = compile_source(mvp_a_source, stop_after=stage)
    assert getattr(result, reached) is not None
    assert getattr(result, not_reached) is None


def test_unknown_stage_is_rejected(mvp_a_source: str) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        compile_source(mvp_a_source, stop_after="nonsense")


def test_syntax_error_raises(mvp_a_source: str) -> None:
    with pytest.raises(SlanqError):
        compile_source("qbool q = ;")


def test_semantic_error_stops_before_codegen() -> None:
    result = compile_source("H(unknown);")
    assert result.diagnostics.has_errors
    assert result.ir is None
    assert result.qiskit_source is None


def test_source_name_reaches_the_header(mvp_a_source: str) -> None:
    result = compile_source(mvp_a_source, source_name="demo.slanq")
    assert result.qiskit_source is not None
    assert "from demo.slanq" in result.qiskit_source
