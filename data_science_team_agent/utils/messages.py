"""Message utilities for handling conversation messages.

Provides helper functions to extract content from different
types of messages in conversation sequences.
"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # type: ignore[import]


def get_last_user_message_content(messages: Sequence[BaseMessage]) -> str:
    """Get the content of the last user message from a sequence of messages."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def get_last_ai_message_content(messages: Sequence[BaseMessage]) -> str:
    """Get the content of the last AI message from a sequence of messages."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content
    return ""


def extract_user_instructions(messages: Sequence[BaseMessage]) -> str:
    """Extract user instructions from messages."""
    user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    return " ".join(user_messages)


def format_messages_for_prompt(messages: Sequence[BaseMessage]) -> str:
    """Format messages for inclusion in a prompt."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")

    return "\n".join(formatted)


def create_message_from_content(content: str, role: str = "user") -> BaseMessage:
    """Create a message from content and role."""
    if role.lower() == "user":
        return HumanMessage(content=content)
    elif role.lower() == "ai":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)  # Default to user


def get_tool_call_names(messages: Sequence[BaseMessage]) -> list[str]:
    """Get names of tool calls from a sequence of messages."""
    tool_names = []
    for message in messages:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                if hasattr(tool_call, "name"):
                    tool_names.append(tool_call.name)
    return tool_names
