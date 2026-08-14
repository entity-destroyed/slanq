"""Driver of the compilation pipeline.

Architectural rule: `analysis.py` never mutates the IR, it only derives facts
about it; mutation is performed exclusively by `passes.py`. This separation is
enforced from the start so that the pipeline can later be moved onto a
Qiskit-style PassManager without restructuring anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lark import ParseTree

from slanq.analysis import analyze
from slanq.ast_nodes import Program
from slanq.codegen import generate_qiskit
from slanq.diagnostics import DiagnosticBag
from slanq.ir import IRModule
from slanq.lowering import lower_to_ir
from slanq.parser import parse_source
from slanq.transformer import transform_to_ast

STAGES = ("parse", "ast", "semantic", "ir", "codegen")


@dataclass
class CompilationResult:
    """Full output of a single compilation run.

    Stages that did not run leave their field as None, either because
    `stop_after` cut the pipeline short or because an earlier stage reported an
    error.
    """

    diagnostics: DiagnosticBag = field(default_factory=DiagnosticBag)
    parse_tree: ParseTree | None = None
    ast: Program | None = None
    ir: IRModule | None = None
    qiskit_source: str | None = None


def compile_source(
    source: str,
    *,
    source_name: str = "<source>",
    stop_after: str | None = None,
) -> CompilationResult:
    """Compile Slanq source code into a Qiskit Python module.

    `stop_after` halts the pipeline after the named stage, which makes the
    output of individual stages inspectable during development and testing.
    """
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError(f"unknown stage {stop_after!r}; expected one of {', '.join(STAGES)}")

    result = CompilationResult()

    result.parse_tree = parse_source(source)
    if stop_after == "parse":
        return result

    result.ast = transform_to_ast(result.parse_tree)
    if stop_after == "ast":
        return result

    analyze(result.ast, result.diagnostics)
    if stop_after == "semantic" or result.diagnostics.has_errors:
        return result

    result.ir = lower_to_ir(result.ast, result.diagnostics)
    if stop_after == "ir" or result.diagnostics.has_errors:
        return result

    result.qiskit_source = generate_qiskit(result.ir, source_name=source_name)
    return result


__all__ = ["STAGES", "CompilationResult", "compile_source"]
