"""Data loader tools agent for automated data loading operations.

Provides specialized agent capabilities for loading and processing
data from various sources with automated workflows.
"""

import operator
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage  # type: ignore[import]

try:
    from langgraph.prebuilt import create_react_agent  # type: ignore[import]
    from langgraph.prebuilt.chat_agent_executor import AgentState  # type: ignore[import]
except ImportError:
    from langchain.agents import AgentState, create_react_agent  # type: ignore[import]
from langgraph.graph import END, START, StateGraph  # type: ignore[import]
from langgraph.types import Checkpointer  # type: ignore[import]

from data_science_team_agent.templates import BaseAgent
from data_science_team_agent.tools.data_loader import (
    get_file_info,
    list_directory_contents,
    list_directory_recursive,
    load_directory,
    load_file,
    search_files_by_pattern,
)
from data_science_team_agent.utils.messages import get_tool_call_names
from data_science_team_agent.utils.regex import format_agent_name

AGENT_NAME = "data_loader_tools_agent"

tools = [
    load_directory,
    load_file,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
]


class DataLoaderToolsAgent(BaseAgent):
    """Agent for loading and processing data from various sources."""

    def __init__(
        self,
        model,
        create_react_agent_kwargs: dict | None = None,
        invoke_react_agent_kwargs: dict | None = None,
        checkpointer: Checkpointer | None = None,
        log_tool_calls: bool = True,
    ):
        """Initialize the data loader tools agent.

        Args:
            model: The language model to use.
            create_react_agent_kwargs: Additional kwargs for creating React agent. Defaults to None.
            invoke_react_agent_kwargs: Additional kwargs for invoking React agent. Defaults to None.
            checkpointer: Checkpointer for state management. Defaults to None.
            log_tool_calls: Whether to log tool calls. Defaults to True.
        """
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
            "log_tool_calls": log_tool_calls,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_data_loader_tools_agent(**self._params)


def make_data_loader_tools_agent(
    model,
    create_react_agent_kwargs: dict | None = None,
    invoke_react_agent_kwargs: dict | None = None,
    checkpointer: Checkpointer | None = None,
    log_tool_calls: bool = True,
):
    """Create a data loader tools agent.

    Args:
        model: The language model to use.
        create_react_agent_kwargs: Additional kwargs for creating React agent. Defaults to None.
        invoke_react_agent_kwargs: Additional kwargs for invoking React agent. Defaults to None.
        checkpointer: Checkpointer for state management. Defaults to None.
        log_tool_calls: Whether to log tool calls. Defaults to True.

    Returns:
        Compiled data loader tools agent graph.
    """
    llm = model

    class GraphState(AgentState):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def route_to_tools(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        if log_tool_calls:
            tool_names = get_tool_call_names(state.get("messages", []))
            if tool_names:
                print(f"    * Tools called: {', '.join(tool_names)}")

        return {"messages": state.get("messages", [])}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("route_to_tools", route_to_tools)
    workflow.add_edge(START, "route_to_tools")
    workflow.add_edge("route_to_tools", END)

    # Create React agent with tools
    agent = create_react_agent(llm, tools, **create_react_agent_kwargs)  # type: ignore[arg-type]

    # Combine with our workflow
    compiled_agent = workflow.compile()
    compiled_agent.agent = agent

    return compiled_agent
