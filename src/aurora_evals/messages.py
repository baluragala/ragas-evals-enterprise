"""Convert LangChain 1.x message traces into Ragas message objects for agent metrics.

Why not `ragas.integrations.langgraph.convert_to_ragas_messages`?
    In ragas 0.4.3 that helper reads tool calls from `AIMessage.additional_kwargs["tool_calls"]`.
    LangChain 1.x no longer puts them there. It uses the standard `AIMessage.tool_calls` field. With
    the built-in helper every tool call silently disappears and tool metrics score 0. Checking
    that your evaluation plumbing sees what you think it sees is itself an important eval lesson.

Free-text arguments
    Ragas' ToolCallAccuracy compares argument values with exact string matching. That is right
    for IDs ("C-1001", "4242") but unfair for free text: `search_policy(query=...)` can be worded
    a hundred valid ways. We "canonicalise" those arguments to a placeholder on BOTH sides
    (prediction and reference), so the metric checks *that* the tool was called, not the wording.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage as LCHumanMessage
from langchain_core.messages import ToolMessage as LCToolMessage
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

# tool name → argument names whose exact wording should not be graded
FREE_TEXT_ARGS: dict[str, set[str]] = {
    "search_policy": {"query"},
    "escalate_to_human": {"summary"},
}
PLACEHOLDER = "<free-text>"


def _text(content) -> str:
    """LangChain content can be a string or a list of content blocks; Ragas needs a string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def mask_args(name: str, args: dict, free_text_args: dict[str, set[str]] | None = None) -> dict:
    free = (free_text_args if free_text_args is not None else FREE_TEXT_ARGS).get(name, set())
    return {k: (PLACEHOLDER if k in free else v) for k, v in args.items()}


def to_ragas_tool_calls(
    calls: list[dict], mask: bool = True, free_text_args: dict[str, set[str]] | None = None
) -> list[ToolCall]:
    """[{"name": ..., "args": {...}}, ...] → [ragas ToolCall, ...] (used for reference calls)."""
    return [
        ToolCall(
            name=c["name"],
            args=mask_args(c["name"], c.get("args", {}), free_text_args) if mask else c.get("args", {}),
        )
        for c in calls
    ]


def to_ragas_messages(
    messages: list[BaseMessage], mask: bool = True, free_text_args: dict[str, set[str]] | None = None
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """LangChain message trace → Ragas message trace. System messages are dropped."""
    converted: list[HumanMessage | AIMessage | ToolMessage] = []
    for m in messages:
        if isinstance(m, LCHumanMessage):
            converted.append(HumanMessage(content=_text(m.content)))
        elif isinstance(m, LCAIMessage):
            converted.append(
                AIMessage(
                    content=_text(m.content),
                    tool_calls=to_ragas_tool_calls(
                        [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls],
                        mask=mask,
                        free_text_args=free_text_args,
                    )
                    or None,
                )
            )
        elif isinstance(m, LCToolMessage):
            converted.append(ToolMessage(content=_text(m.content)))
    return converted
