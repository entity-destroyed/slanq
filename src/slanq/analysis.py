from __future__ import annotations

from slanq.ast_nodes import Declaration, Name, Program
from slanq.diagnostics import DiagnosticBag
from slanq.visitor import NodeVisitor

BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg",
        "RX", "RY", "RZ",
        "CX", "CY", "CZ", "CH", "SWAP", "CCX",
        "phase",
        "measure",
    }
)

Scope = dict[str, Declaration]


def analyze(ast: Program, bag: DiagnosticBag) -> None:
    scope = _build_symbol_table(ast, bag)
    _resolve_names(ast, bag, scope)
    _check_types(ast, bag)
    _check_affine(ast, bag)


def _build_symbol_table(ast: Program, bag: DiagnosticBag) -> Scope:
    scope: Scope = {}
    for stmt in ast.statements:
        if not isinstance(stmt, Declaration):
            continue
        if stmt.name in scope:
            bag.error(
                f"duplicate declaration of '{stmt.name}'",
                line=stmt.span.start_line,
                column=stmt.span.start_col,
            )
            continue
        scope[stmt.name] = stmt
    return scope


class _NameResolver(NodeVisitor):
    def __init__(self, scope: Scope, bag: DiagnosticBag) -> None:
        self.scope = scope
        self.bag = bag

    def visit_Name(self, node: Name) -> None:
        if node.name not in BUILTIN_NAMES:
            declaration = self.scope.get(node.name)
            if declaration is None:
                self.bag.error(
                    f"undefined name '{node.name}'",
                    line=node.span.start_line,
                    column=node.span.start_col,
                )
            else:
                node.resolved_symbol = declaration
        self.generic_visit(node)


def _resolve_names(ast: Program, bag: DiagnosticBag, scope: Scope) -> None:
    _NameResolver(scope, bag).visit(ast)


def _check_types(ast: Program, bag: DiagnosticBag) -> None:
    pass


def _check_affine(ast: Program, bag: DiagnosticBag) -> None:
    pass


__all__ = ["BUILTIN_NAMES", "analyze"]
