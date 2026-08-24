from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator

import pytest
from lark import Lark, Tree

from slanq.ast_nodes import (
    Assign,
    AugAssign,
    BinaryOp,
    Block,
    Call,
    ClassicalDecl,
    ExprStatement,
    For,
    If,
    Index,
    IntType,
    Literal,
    Name,
    Node,
    ParamArrayDecl,
    ParamDecl,
    PostUpdate,
    ProbList,
    ProcessDef,
    Program,
    QBoolType,
    QIf,
    QIntType,
    QuantumDecl,
    UnaryOp,
    While,
)
from slanq.diagnostics import SlanqError
from slanq.transformer import SlanqTransformer

BuildAst = Callable[[str], Program]


def _walk(value: object) -> Iterator[object]:
    """Every reachable value, including raw Lark trees the transformer left behind."""
    yield value
    if isinstance(value, Tree):
        return
    if isinstance(value, Node):
        for f in dataclasses.fields(value):
            yield from _walk(getattr(value, f.name))
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _raw_trees(ast: Program) -> list[str]:
    return [str(value.data) for value in _walk(ast) if isinstance(value, Tree)]


def test_qbool_decl_becomes_quantum_decl(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qbool q = false;").statements
    assert isinstance(declaration, QuantumDecl)
    assert declaration.name == "q"
    assert isinstance(declaration.declared_type, QBoolType)
    assert isinstance(declaration.initializer, Literal)
    assert declaration.initializer.value is False


def test_qint_decl_carries_its_width(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qint<3> a = 0;").statements
    assert isinstance(declaration, QuantumDecl)
    assert declaration.declared_type == QIntType(size=3)


def test_empty_prob_list_means_equal_superposition(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qint<2> b = [];").statements
    assert isinstance(declaration, QuantumDecl)
    assert isinstance(declaration.initializer, ProbList)
    assert declaration.initializer.probabilities == []


def test_prob_list_keeps_its_values(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qint<2> c = [0, 0.5, 0.5, 0];").statements
    assert isinstance(declaration, QuantumDecl)
    assert isinstance(declaration.initializer, ProbList)
    assert declaration.initializer.probabilities == [0.0, 0.5, 0.5, 0.0]


def test_param_declarations(build_ast: BuildAst) -> None:
    scalar, array = build_ast("param float theta;\nparam int gamma[4];").statements
    assert isinstance(scalar, ParamDecl)
    assert scalar.name == "theta"
    assert isinstance(array, ParamArrayDecl)
    assert array.name == "gamma"
    assert array.size == 4


def test_gate_call_becomes_expr_statement(build_ast: BuildAst) -> None:
    (statement,) = build_ast("H(q);").statements
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    assert statement.expr.callee.name == "H"
    (argument,) = statement.expr.args
    assert isinstance(argument, Name)
    assert argument.name == "q"


def test_call_without_arguments_has_empty_arg_list(build_ast: BuildAst) -> None:
    (statement,) = build_ast("H();").statements
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    assert statement.expr.args == []


def test_indexing_becomes_index_node(build_ast: BuildAst) -> None:
    (statement,) = build_ast("X(a[1]);").statements
    assert isinstance(statement, ExprStatement)
    assert isinstance(statement.expr, Call)
    (argument,) = statement.expr.args
    assert isinstance(argument, Index)
    assert argument.base.name == "a"
    assert isinstance(argument.index, Literal)
    assert argument.index.value == 1


def test_assignment_and_compound_assignment(build_ast: BuildAst) -> None:
    plain, compound = build_ast("x = 1;\na += b;").statements
    assert isinstance(plain, Assign)
    assert isinstance(compound, AugAssign)
    assert compound.op == "+="


def test_classical_decl_with_measurement(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("int result = measure(q);").statements
    assert isinstance(declaration, ClassicalDecl)
    assert declaration.name == "result"
    assert isinstance(declaration.declared_type, IntType)
    assert isinstance(declaration.initializer, Call)
    assert declaration.initializer.callee.name == "measure"


def test_branches_and_loops(build_ast: BuildAst) -> None:
    source = "if(x) { X(q); }\nqif(y) { X(q); }\nwhile(z) { X(q); }"
    classical, quantum, loop = build_ast(source).statements
    assert isinstance(classical, If)
    assert isinstance(quantum, QIf)
    assert isinstance(loop, While)
    assert isinstance(classical.body, Block)
    assert len(classical.body.statements) == 1


def test_for_loop_parts(build_ast: BuildAst) -> None:
    (loop,) = build_ast("for(int i = 0; i < 4; i++) { X(q); }").statements
    assert isinstance(loop, For)
    assert isinstance(loop.init, ClassicalDecl)
    assert loop.init.name == "i"
    assert isinstance(loop.condition, BinaryOp)
    assert loop.condition.op == "<"
    assert isinstance(loop.update, PostUpdate)
    assert loop.update.op == "++"


def test_process_definition(build_ast: BuildAst) -> None:
    (process,) = build_ast("process add(qint x, qint y) { x += y; }").statements
    assert isinstance(process, ProcessDef)
    assert process.name == "add"
    assert [parameter.name for parameter in process.params] == ["x", "y"]
    assert isinstance(process.body, Block)


def test_process_without_parameters(build_ast: BuildAst) -> None:
    (process,) = build_ast("process f() { X(q); }").statements
    assert isinstance(process, ProcessDef)
    assert process.params == []


def test_binary_operator_keeps_its_symbol(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qint<3> c = a + b;").statements
    assert isinstance(declaration, QuantumDecl)
    assert isinstance(declaration.initializer, BinaryOp)
    assert declaration.initializer.op == "+"


def test_unary_operator_keeps_its_symbol(build_ast: BuildAst) -> None:
    (branch,) = build_ast("if(!x) { X(q); }").statements
    assert isinstance(branch, If)
    assert isinstance(branch.condition, UnaryOp)
    assert branch.condition.op == "!"


def test_precedence_layers_are_flattened(build_ast: BuildAst) -> None:
    """`a + b * c` must nest by precedence, not by the grammar's helper rules."""
    (declaration,) = build_ast("int x = a + b * c;").statements
    assert isinstance(declaration, ClassicalDecl)
    outer = declaration.initializer
    assert isinstance(outer, BinaryOp)
    assert outer.op == "+"
    assert isinstance(outer.right, BinaryOp)
    assert outer.right.op == "*"


def test_mvp_a_full_source(build_ast: BuildAst, mvp_a_source: str) -> None:
    statements = build_ast(mvp_a_source).statements
    assert len(statements) == 3
    assert isinstance(statements[0], QuantumDecl)
    assert isinstance(statements[1], ExprStatement)
    assert isinstance(statements[2], ClassicalDecl)


def test_hello_example_leaves_no_raw_parse_tree(build_ast: BuildAst, hello_source: str) -> None:
    """Any rule the transformer misses would survive as a lark.Tree and then be
    silently skipped by every visitor."""
    assert _raw_trees(build_ast(hello_source)) == []


def test_unhandled_grammar_rule_fails_loudly() -> None:
    """The test above only covers rules hello.slanq happens to use; this guard
    covers every rule added to the grammar from now on."""
    parser = Lark(
        r"start: mystery" "\n" r"mystery: NAME" "\n" r"NAME: /[a-z]+/" "\n" r"%ignore /\s+/",
        propagate_positions=True,
    )
    with pytest.raises(SlanqError, match="no AST transformation"):
        SlanqTransformer().transform(parser.parse("abc"))


def test_spans_are_populated(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qbool q = false;").statements
    assert declaration.span.start_line == 1
    assert declaration.span.start_col >= 1


def test_number_literal_stays_an_int(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("int x = 42;").statements
    assert isinstance(declaration, ClassicalDecl)
    assert isinstance(declaration.initializer, Literal)
    assert declaration.initializer.value == 42
    assert isinstance(declaration.initializer.value, int)


def test_boolean_true_literal(build_ast: BuildAst) -> None:
    (declaration,) = build_ast("qbool q = true;").statements
    assert isinstance(declaration, QuantumDecl)
    assert isinstance(declaration.initializer, Literal)
    assert declaration.initializer.value is True
