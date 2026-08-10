"""Tests for the diagnostics module (diagnostics.py)."""

from __future__ import annotations

import pytest

from slanq.diagnostics import DiagnosticBag, Severity, SlanqError


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
