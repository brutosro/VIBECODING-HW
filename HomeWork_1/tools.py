"""Local tools the LLM can call, plus their JSON schemas and a dispatcher."""

import ast
import json
import operator
from collections.abc import Callable
from typing import Any

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate an arithmetic expression (no eval)."""
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _evaluate(tree.body)}


def count_text(text: str, substring: str | None = None) -> dict[str, Any]:
    """Count characters, words, and optionally case-insensitive occurrences of a substring."""
    result: dict[str, Any] = {
        "characters": len(text),
        "words": len(text.split()),
    }
    if substring:
        result["substring"] = substring
        result["occurrences"] = text.lower().count(substring.lower())
    return result


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression with + - * / // % ** and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(17 * 23) / 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_text",
            "description": (
                "Count characters and words in a text. If 'substring' is given, "
                "also count its case-insensitive occurrences (e.g. letters in a word)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to analyse."},
                    "substring": {
                        "type": "string",
                        "description": "Optional letter or substring to count, e.g. 'r'.",
                    },
                },
                "required": ["text"],
            },
        },
    },
]

_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate": calculate,
    "count_text": count_text,
}


def run_tool(name: str, raw_arguments: str) -> str:
    """Execute a tool call and return a JSON string for the 'tool' message.

    Errors are returned to the model (not raised) so it can correct itself.
    """
    if name not in _REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        arguments = json.loads(raw_arguments or "{}")
        return json.dumps(_REGISTRY[name](**arguments))
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as error:
        return json.dumps({"error": f"{type(error).__name__}: {error}"})
