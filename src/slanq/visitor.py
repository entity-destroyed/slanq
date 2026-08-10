from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

from slanq.ast_nodes import Node


def iter_child_nodes(node: Node) -> Iterator[Node]:
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        if isinstance(value, Node):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Node):
                    yield item


class NodeVisitor:
    def visit(self, node: Node) -> Any:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is not None:
            return method(node)
        return self.generic_visit(node)

    def generic_visit(self, node: Node) -> Any:
        for child in iter_child_nodes(node):
            self.visit(child)
        return None


__all__ = ["NodeVisitor", "iter_child_nodes"]
