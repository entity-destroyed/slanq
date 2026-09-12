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
    """Lowering still runs after a semantic error -- it independently
    validates its own concerns, so skipping it would hide diagnostics behind
    whichever error analysis happened to find first. Only codegen requires a
    fully clean run."""
    result = compile_source("H(unknown);")
    assert result.diagnostics.has_errors
    assert result.ir is not None
    assert result.qiskit_source is None


def test_lowering_runs_despite_a_prior_analysis_error() -> None:
    """An analysis error must not hide an independent lowering-phase error --
    each phase validates its own concerns, and gating lowering behind
    analysis would silently drop diagnostics that have nothing to do with
    the analysis error itself."""
    result = compile_source("qbool q = false; H(unknown); while(true) { X(q); }")
    messages = [d.message for d in result.diagnostics.errors]
    assert any("undefined name" in message for message in messages)
    assert any("while loop" in message for message in messages)


def test_reserved_name_never_reaches_codegen() -> None:
    """`circuit` used to compile clean and only fail at generated-file
    runtime (IndexError, from shadowing the compiler's own variable) --
    this must now be caught as a compile-time error instead."""
    result = compile_source("qbool circuit = false;\nH(circuit);\n")
    assert result.diagnostics.has_errors
    assert result.qiskit_source is None


def test_source_name_reaches_the_header(mvp_a_source: str) -> None:
    result = compile_source(mvp_a_source, source_name="demo.slanq")
    assert result.qiskit_source is not None
    assert "from demo.slanq" in result.qiskit_source
