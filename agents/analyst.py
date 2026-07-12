
from __future__ import annotations

import json
import re

import structlog

from agents.tools import tool_registry
from models.llm import Message, get_llm

logger = structlog.get_logger()

ANALYST_SYSTEM = """You are an expert research analyst. Your job is to:
1. Analyze the retrieved context for the given query
2. Identify the most relevant facts, insights, and patterns
3. Decide if any tools are needed (calculator for math, python_repl for data analysis)
4. Identify any gaps or contradictions in the context

If you need to use a tool, respond with a JSON action:
{"action": "tool", "tool": "calculator", "input": "expression"}
{"action": "tool", "tool": "python_repl", "input": "code"}

Otherwise, respond with your analysis directly starting with "ANALYSIS:"

Be concise and structured."""

ANALYST_PROMPT = """Query: {query}
Query Type: {query_type}

Retrieved Context:
---
{context}
---

Analyze the above context in relation to the query. Extract key insights,
identify what directly answers the question, and note any gaps."""


class AnalystAgent:
    """
    Analyzes retrieved context and optionally uses tools for computation.
    Supports up to 3 tool calls per analysis cycle.
    """

    MAX_TOOL_CALLS = 3

    def __init__(self):
        self._llm = get_llm(temperature=0.1)

    def analyze(
        self,
        query: str,
        context: str,
        query_type: str = "factual",
    ) -> tuple[str, list[dict]]:
        """
        Returns (analysis_text, tool_calls_made).
        """
        tool_calls: list[dict] = []
        tool_outputs: list[str] = []

        messages = [
            Message(
                "user",
                ANALYST_PROMPT.format(
                    query=query,
                    query_type=query_type,
                    context=context[:4000],  # respect context window
                ),
            )
        ]

        # Agentic tool-use loop (max 3 iterations)
        for _ in range(self.MAX_TOOL_CALLS):
            response = self._llm.chat(messages, system=ANALYST_SYSTEM)
            content = response.content.strip()

            # Check if LLM wants to use a tool
            tool_call = self._extract_tool_call(content)
            if tool_call:
                tool_name = tool_call.get("tool", "")
                tool_input = tool_call.get("input", "")

                # Execute tool
                if tool_name == "calculator":
                    result = tool_registry.call("calculator", expression=tool_input)
                elif tool_name == "python_repl":
                    result = tool_registry.call("python_repl", code=tool_input)
                elif tool_name == "web_search":
                    result = tool_registry.call("web_search", query=tool_input)
                else:
                    break

                tool_calls.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": result.output,
                    "success": result.success,
                })

                # Feed result back into conversation
                messages.append(Message("assistant", content))
                messages.append(
                    Message(
                        "user",
                        f"Tool result from {tool_name}:\n{result.output}\n\nContinue your analysis.",
                    )
                )
                continue

            # If no tool needed, return analysis
            analysis = content.replace("ANALYSIS:", "").strip()
            if tool_outputs:
                analysis = f"Tool findings: {'; '.join(tool_outputs)}\n\n{analysis}"

            return analysis, tool_calls

        # Fallback: return whatever we have
        final_response = self._llm.chat(
            messages + [Message("user", "Summarize your analysis now.")],
            system=ANALYST_SYSTEM,
        )
        return final_response.content, tool_calls

    @staticmethod
    def _extract_tool_call(content: str) -> dict | None:
        """Parse JSON tool call from LLM output."""
        try:
            match = re.search(r'\{[^{}]*"action"\s*:\s*"tool"[^{}]*\}', content)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None
