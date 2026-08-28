from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from slanq.ast_nodes import (
    Call,
    ClassicalDecl,
    ExprStatement,
    IntType,
    Literal,
    Name,
    Program,
    QBoolType,
    QuantumDecl,
    Span,
)
from slanq.ir import (
    ClbitRef,
    GateOp,
    InitOp,
    IRBlock,
    IRModule,
    MeasurementOp,
    QubitBit,
    QubitRef,
)
from slanq.parser import parse_source
from slanq.transformer import transform_to_ast

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def span() -> Span:
    return Span(start_line=1, start_col=1, end_line=1, end_col=2)


@pytest.fixture
def build_ast() -> Callable[[str], Program]:
    def _build_ast(source: str) -> Program:
        return transform_to_ast(parse_source(source))

    return _build_ast


@pytest.fixture
def mvp_a_source() -> str:
    return (EXAMPLES / "mvp_a.slanq").read_text(encoding="utf-8")


@pytest.fixture
def mvp_b_source() -> str:
    return (EXAMPLES / "mvp_b.slanq").read_text(encoding="utf-8")


@pytest.fixture
def hello_source() -> str:
    return (EXAMPLES / "hello.slanq").read_text(encoding="utf-8")


@pytest.fixture
def mvp_a_ast(span: Span) -> Program:
    """MVP (a) assembled by hand, so consumers can be tested without the parser."""
    return Program(
        span=span,
        statements=[
            QuantumDecl(
                span=span,
                name="q",
                declared_type=QBoolType(),
                initializer=Literal(span=span, value=False),
            ),
            ExprStatement(
                span=span,
                expr=Call(
                    span=span,
                    callee=Name(span=span, name="H"),
                    args=[Name(span=span, name="q")],
                ),
            ),
            ClassicalDecl(
                span=span,
                name="result",
                declared_type=IntType(),
                initializer=Call(
                    span=span,
                    callee=Name(span=span, name="measure"),
                    args=[Name(span=span, name="q")],
                ),
            ),
        ],
    )


@pytest.fixture
def mvp_a_ir(span: Span) -> IRModule:
    """MVP (a) assembled by hand, so codegen can be tested without the lowering."""
    qubit = QubitRef(name="q", size=1)
    clbit = ClbitRef(name="result", size=1)
    return IRModule(
        qubits=[qubit],
        clbits=[clbit],
        body=IRBlock(
            ops=[
                InitOp(span=span, target=qubit, value=False),
                GateOp(span=span, name="H", targets=[QubitBit(ref=qubit, index=0)]),
                MeasurementOp(span=span, source=qubit, target=clbit),
            ]
        ),
    )
