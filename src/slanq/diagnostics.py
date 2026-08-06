"""Unified handling of compiler messages (errors and warnings).

Besides reporting errors, the compiler also emits warnings for patterns that
work in theory but are questionable on today's noisy hardware (for example too
many qubits, or an expensive state preparation). Errors and warnings therefore
share a single channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Diagnostic:
    """A single compiler message with an optional source position."""

    severity: Severity
    message: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        where = ""
        if self.line is not None:
            where = f"{self.line}:{self.column}: " if self.column is not None else f"{self.line}: "
        return f"{where}{self.severity.value}: {self.message}"


class SlanqError(Exception):
    """An error that aborts compilation."""


@dataclass
class DiagnosticBag:
    """Messages collected during a single compilation run."""

    items: list[Diagnostic] = field(default_factory=list)

    def error(self, message: str, line: int | None = None, column: int | None = None) -> None:
        self.items.append(Diagnostic(Severity.ERROR, message, line, column))

    def warning(self, message: str, line: int | None = None, column: int | None = None) -> None:
        self.items.append(Diagnostic(Severity.WARNING, message, line, column))

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.items if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.items if d.severity is Severity.WARNING]

    @property
    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.items)

    def raise_if_errors(self) -> None:
        """Abort compilation if any error (not merely a warning) was collected."""
        if self.has_errors:
            joined = "\n".join(str(d) for d in self.errors)
            raise SlanqError(f"Compilation failed:\n{joined}")

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
