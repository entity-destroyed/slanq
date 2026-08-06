"""Driver of the compilation pipeline.

The stages follow the "Compilation" chapter of the documentation:

    1. Build the abstract syntax tree (AST)
    2. Build the symbol table, perform type checking
    3. Produce the intermediate representation (IR)
    4. Quantum analysis (entanglement graph, lifetime analysis)
    5. Optimize the intermediate representation
    6. Generate Qiskit code

Important architectural rule: the `analysis` package NEVER mutates the IR, it
only derives facts about it; mutation is performed exclusively by the `passes`
package. This separation is enforced from the start so that the pipeline can
later be moved onto a Qiskit-style PassManager mechanically, without rewriting
anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slanq.diagnostics import DiagnosticBag
from slanq.parser import parse_source


@dataclass
class CompilationResult:
    """Full output of a single compilation run."""

    diagnostics: DiagnosticBag = field(default_factory=DiagnosticBag)
    parse_tree: Any | None = None
    ast: Any | None = None
    ir: Any | None = None
    circuit: Any | None = None


def compile_source(source: str, *, stop_after: str | None = None) -> CompilationResult:
    """Compile Slanq source code.

    The `stop_after` parameter halts the pipeline after a given stage
    ("parse", "ast", "semantic", "ir", "analysis", "optimize"). This makes the
    output of individual stages inspectable during development and testing.
    """
    result = CompilationResult()

    # Stage 1 - syntactic analysis
    result.parse_tree = parse_source(source)
    if stop_after == "parse":
        return result

    # Later stages are still being implemented.
    # Their skeletons live in the semantic/, ir/, analysis/, passes/ and
    # codegen/ packages.
    raise NotImplementedError(
        "Stages beyond syntactic analysis are not implemented yet. "
        "Use stop_after='parse'."
    )


__all__ = ["CompilationResult", "compile_source"]
