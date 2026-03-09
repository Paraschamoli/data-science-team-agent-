from typing_extensions import Any, Optional, Annotated, Sequence, List, Dict
import operator
import pandas as pd
import os
from langchain_core.messages import BaseMessage, AIMessage
try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState
except ImportError:
    from langchain.agents import create_react_agent, AgentState
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from data_science_team_agent.templates import BaseAgent
from data_science_team_agent.utils.regex import format_agent_name
from data_science_team_agent.tools.data_loader import (
    load_directory,
    load_file,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
)
from data_science_team_agent.utils.messages import get_tool_call_names

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
    def __init__(
        self,
        model,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
        log_tool_calls: bool = True,
    ):
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
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
    checkpointer: Optional[Checkpointer] = None,
    log_tool_calls: bool = True,
):
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
    agent = create_react_agent(llm, tools, **create_react_agent_kwargs)
    
    # Combine with our workflow
    compiled_agent = workflow.compile()
    compiled_agent.agent = agent
    
    return compiled_agent
