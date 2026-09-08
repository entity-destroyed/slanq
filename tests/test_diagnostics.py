"""Tests for the diagnostics module (diagnostics.py)."""

from __future__ import annotations

import pytest

from slanq.diagnostics import Diagnostic, DiagnosticBag, Severity, SlanqError


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


def test_an_identical_message_at_an_identical_position_is_kept_once() -> None:
    bag = DiagnosticBag()
    bag.error("a += a", line=4, column=8)
    bag.error("a += a", line=4, column=8)
    assert len(bag.items) == 1


def test_the_same_message_at_another_position_is_kept() -> None:
    bag = DiagnosticBag()
    bag.error("a += a", line=4, column=8)
    bag.error("a += a", line=9, column=8)
    bag.error("a += a", line=4, column=2)
    bag.error("a += a")
    assert len(bag.items) == 4


def test_an_error_and_a_warning_with_one_text_are_both_kept() -> None:
    bag = DiagnosticBag()
    bag.error("state preparation is expensive", line=1, column=1)
    bag.warning("state preparation is expensive", line=1, column=1)
    assert len(bag.errors) == 1
    assert len(bag.warnings) == 1


def test_a_bag_built_from_existing_items_still_deduplicates() -> None:
    existing = Diagnostic(Severity.ERROR, "a += a", 4, 8)
    bag = DiagnosticBag(items=[existing])
    bag.error("a += a", line=4, column=8)
    assert len(bag.items) == 1
