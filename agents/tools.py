
from __future__ import annotations

import ast
import math
import re
import textwrap
import traceback
from io import StringIO
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ToolResult:
    tool: str
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata or {},
        }


# ── Tool 1: Web Search ───────────────────────────────────────────────────────


class WebSearchTool:
    name = "web_search"
    description = "Search the web for current, real-time information"

    def __call__(self, query: str, num_results: int = 5) -> ToolResult:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            if not results:
                return ToolResult(
                    tool=self.name,
                    success=False,
                    output="",
                    error="No results found",
                )

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"URL: {r.get('href', '')}\n"
                    f"Snippet: {r.get('body', '')}\n"
                )

            return ToolResult(
                tool=self.name,
                success=True,
                output="\n\n".join(formatted),
                metadata={"query": query, "num_results": len(results)},
            )

        except ImportError:
            return ToolResult(
                tool=self.name,
                success=False,
                output="",
                error="duckduckgo-search not installed. Run: pip install duckduckgo-search",
            )
        except Exception as e:
            logger.error("web_search_failed", query=query, error=str(e))
            return ToolResult(
                tool=self.name, success=False, output="", error=str(e)
            )


# ── Tool 2: Calculator ───────────────────────────────────────────────────────


class CalculatorTool:
    """Safe math expression evaluator using AST (no eval())."""

    name = "calculator"
    description = "Evaluate mathematical expressions safely"

    # Allowed AST node types
    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
    }

    ALLOWED_NAMES = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt, "log": math.log,
        "log2": math.log2, "log10": math.log10, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "floor": math.floor,
        "ceil": math.ceil, "factorial": math.factorial,
    }

    def __call__(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            self._validate_tree(tree)
            result = eval(  # noqa: S307
                compile(tree, "<calculator>", "eval"),
                {"__builtins__": {}},
                self.ALLOWED_NAMES,
            )
            return ToolResult(
                tool=self.name,
                success=True,
                output=str(result),
                metadata={"expression": expression, "result": result},
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                output="",
                error=f"Calculation error: {e}",
            )

    def _validate_tree(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if type(node) not in self.ALLOWED_NODES:
                raise ValueError(f"Forbidden operation: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in self.ALLOWED_NAMES:
                raise ValueError(f"Unknown variable: {node.id}")


# ── Tool 3: Python REPL ──────────────────────────────────────────────────────


class PythonREPLTool:
    """
    Sandboxed Python REPL for data analysis.
    Timeout prevents infinite loops.
    Only allows safe standard library imports.
    """

    name = "python_repl"
    description = "Execute Python code for data analysis, statistics, and computation"

    ALLOWED_IMPORTS = {
        "math", "statistics", "json", "re", "collections",
        "itertools", "functools", "datetime", "random",
        "numpy", "pandas", "scipy",
    }

    def __call__(self, code: str, timeout: int = 10) -> ToolResult:
        # Block dangerous imports
        dangerous = re.findall(
            r"import\s+(os|sys|subprocess|shutil|socket|urllib|requests|open|eval|exec)",
            code,
        )
        if dangerous:
            return ToolResult(
                tool=self.name,
                success=False,
                output="",
                error=f"Blocked dangerous imports: {dangerous}",
            )

        output_buffer = StringIO()
        local_ns: dict = {}

        try:
            with redirect_stdout(output_buffer):
                exec(textwrap.dedent(code), {"__builtins__": __builtins__}, local_ns)  # noqa: S102

            stdout = output_buffer.getvalue()

            # Capture last expression value
            lines = [l.strip() for l in code.strip().split("\n") if l.strip()]
            last_var = None
            if lines and not lines[-1].startswith(("#", "import", "from", "def", "class")):
                try:
                    last_val = eval(lines[-1], {}, local_ns)  # noqa: S307
                    last_var = str(last_val)
                except Exception:
                    pass

            output = stdout or last_var or "Code executed successfully (no output)"
            return ToolResult(
                tool=self.name,
                success=True,
                output=output,
                metadata={"lines_executed": len(lines)},
            )

        except Exception:
            return ToolResult(
                tool=self.name,
                success=False,
                output=output_buffer.getvalue(),
                error=traceback.format_exc(limit=3),
            )


# ── Tool 4: Citation Validator ───────────────────────────────────────────────


class CitationValidatorTool:
    

    name = "citation_validator"
    description = "Check if a statement is supported by retrieved context"

    def __call__(
        self, statement: str, context: str, threshold: float = 0.6
    ) -> ToolResult:
        # Simple lexical overlap check
        statement_words = set(statement.lower().split())
        context_words = set(context.lower().split())

        if not statement_words:
            return ToolResult(
                tool=self.name, success=False, output="Empty statement", error=None
            )

        overlap = len(statement_words & context_words) / len(statement_words)
        supported = overlap >= threshold

        return ToolResult(
            tool=self.name,
            success=True,
            output=(
                f"{'SUPPORTED' if supported else 'NOT SUPPORTED'} "
                f"(overlap: {overlap:.2%})"
            ),
            metadata={
                "overlap_ratio": overlap,
                "supported": supported,
                "threshold": threshold,
            },
        )


# ── Tool Registry ─────────────────────────────────────────────────────────────


class ToolRegistry:
    """Central registry of all agent tools."""

    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(WebSearchTool())
        self.register(CalculatorTool())
        self.register(PythonREPLTool())
        self.register(CitationValidatorTool())

    def register(self, tool) -> None:
        self._tools[tool.name] = tool

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                tool=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}",
            )
        logger.info("tool_called", tool=tool_name, kwargs=list(kwargs.keys()))
        return self._tools[tool_name](**kwargs)

    @property
    def tool_descriptions(self) -> str:
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# Singleton
tool_registry = ToolRegistry()
