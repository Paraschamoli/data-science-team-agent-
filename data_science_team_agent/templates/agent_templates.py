"""Agent template utilities for creating and managing agents.

Provides base classes and utility functions for creating
data science agents with common patterns and workflows.
"""

from typing import Any

import pandas as pd
import sqlalchemy as sql
from langchain_core.runnables import RunnableConfig  # type: ignore[import]
from langgraph.graph import END, StateGraph  # type: ignore[import]
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import]
from langgraph.types import Command  # type: ignore[import]

from data_science_team_agent.parsers.parsers import PythonOutputParser
from data_science_team_agent.utils.regex import (
    add_comments_to_top,
    relocate_imports_inside_function,
)


class BaseAgent(CompiledStateGraph):
    """A generic base class for agents that interact with compiled state graphs."""

    def __init__(self, **params):
        """Initialize the base agent with parameters."""
        self._params = params
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        self.name = self._compiled_graph.name
        self.checkpointer = self._compiled_graph.checkpointer
        self.store = self._compiled_graph.store
        self.output_channels = self._compiled_graph.output_channels
        self.nodes = self._compiled_graph.nodes
        self.stream_mode = self._compiled_graph.stream_mode
        self.builder = self._compiled_graph.builder
        self.channels = self._compiled_graph.channels
        self.input_channels = self._compiled_graph.input_channels
        self.input_schema = self._compiled_graph.input_schema
        self.output_schema = self._compiled_graph.output_schema
        self.debug = self._compiled_graph.debug
        self.interrupt_after_nodes = self._compiled_graph.interrupt_after_nodes
        self.interrupt_before_nodes = self._compiled_graph.interrupt_before_nodes
        self.config = self._compiled_graph.config

    def _make_compiled_graph(self):
        raise NotImplementedError("Subclasses must implement `_make_compiled_graph` method.")

    def update_params(self, **kwargs):
        """Update the agent parameters."""
        self._params.update(kwargs)
        self._compiled_graph = self._make_compiled_graph()

    def __getattr__(self, name: str):
        """Delegate attribute access to the compiled graph."""
        return getattr(self._compiled_graph, name)

    def invoke(self, input_data: dict[str, Any] | Any, config: RunnableConfig | None = None, **kwargs):
        """Invoke the agent with input data."""
        return self._compiled_graph.invoke(input_data, config, **kwargs)

    def ainvoke(self, input_data: dict[str, Any] | Any, config: RunnableConfig | None = None, **kwargs):
        """Asynchronously invoke the agent with input data."""
        return self._compiled_graph.ainvoke(input_data, config, **kwargs)

    def stream(self, input_data: dict[str, Any] | Any, config: RunnableConfig | None = None, **kwargs):
        """Stream the agent execution."""
        return self._compiled_graph.stream(input_data, config, **kwargs)

    def astream(self, input_data: dict[str, Any] | Any, config: RunnableConfig | None = None, **kwargs):
        """Asynchronously stream the agent execution."""
        return self._compiled_graph.astream(input_data, config, **kwargs)

    def get_state(self, config: RunnableConfig | None = None, **kwargs):
        """Get the current state of the agent."""
        return self._compiled_graph.get_state(config, **kwargs)

    def update_state(self, state: dict[str, Any], config: RunnableConfig | None = None, **kwargs):
        """Update the state of the agent."""
        return self._compiled_graph.update_state(state, config, **kwargs)

    def get_graph(self, **kwargs):
        """Get the agent graph."""
        return self._compiled_graph.get_graph(**kwargs)

    def draw_mermaid_png(self, **kwargs):
        """Draw the agent graph as a Mermaid PNG."""
        return self._compiled_graph.draw_mermaid_png(**kwargs)

    def get_response(self):
        """Get the agent response."""
        return self.response


def node_func_human_review(
    state, prompt_text, yes_goto, no_goto, user_instructions_key, recommended_steps_key, code_snippet_key
):
    """Handle human review of agent recommendations."""
    user_instructions = state.get(user_instructions_key, "")
    recommended_steps = state.get(recommended_steps_key, "")
    code_snippet = state.get(code_snippet_key, "")

    full_prompt = prompt_text.format(
        steps=recommended_steps, user_instructions=user_instructions, code_snippet=code_snippet
    )

    print("    * HUMAN REVIEW")
    print(f"    Prompt: {full_prompt}")

    # For automated execution, we'll skip human review and go to yes_goto
    return Command(goto=yes_goto)


def node_func_fix_agent_code(
    state, code_snippet_key, error_key, llm, prompt_template, agent_name, log, file_path, function_name
):
    """Fix broken agent code."""
    code_snippet = state.get(code_snippet_key, "")
    error = state.get(error_key, "")

    print(f"    * FIX {agent_name.upper()} CODE")

    fix_agent = prompt_template | llm | PythonOutputParser()
    fixed_code = fix_agent.invoke({"function_name": function_name, "code_snippet": code_snippet, "error": error})

    fixed_code = relocate_imports_inside_function(fixed_code)
    fixed_code = add_comments_to_top(fixed_code, agent_name=agent_name)

    return {code_snippet_key: fixed_code}


def node_func_report_agent_outputs(state, keys_to_include):
    """Report final outputs from agent execution."""
    print("    * REPORT AGENT OUTPUTS")

    report = {"report_title": "Agent Outputs", "outputs": {}}

    for key in keys_to_include:
        if key in state:
            report["outputs"][key] = state[key]

    return {"agent_outputs": report}


def node_func_execute_agent_code_on_data(state, function_name_key, data_key, llm, timeout=10, memory_limit_mb=512):
    """Execute agent code on data in sandboxed environment."""
    print(f"    * EXECUTE {function_name_key.upper()} CODE (SANDBOXED)")

    code_snippet = state.get(function_name_key, "")
    data = state.get(data_key, {})

    # Simple sandbox execution (in production, use proper sandbox)
    try:
        local_vars = {}
        exec(code_snippet, globals(), local_vars)  # noqa: S102 - intentional exec for code generation
        func = local_vars.get(function_name_key)

        if func:
            result = func(pd.DataFrame(data))
            return {"data_processed": result.to_dict()}
        else:
            return {"error": f"Function {function_name_key} not found in code"}
    except Exception as e:
        return {"error": str(e)}


def node_func_execute_agent_from_sql_connection(state, sql_query_key, connection_string_key, llm):
    """Execute SQL query from agent."""
    print("    * EXECUTE SQL QUERY")

    sql_query = state.get(sql_query_key, "")
    connection_string = state.get(connection_string_key, "")

    try:
        engine = sql.create_engine(connection_string)
        result = pd.read_sql(sql_query, engine)
        return {"query_result": result.to_dict()}
    except Exception as e:
        return {"error": str(e)}


def node_func_explain_agent_code(state, code_snippet_key, llm, agent_name):
    """Explain agent-generated code."""
    print(f"    * EXPLAIN {agent_name.upper()} CODE")

    code_snippet = state.get(code_snippet_key, "")

    explanation_prompt = f"""
    Explain the following {agent_name} code:

    {code_snippet}

    Provide a clear explanation of what this code does and how it works.
    """

    explanation = llm.invoke(explanation_prompt)
    return {"code_explanation": explanation.content}


def create_coding_agent_graph(nodes, edges, state_class, entry_point="start", checkpointer=None):
    """Create a coding agent graph with specified nodes and edges."""
    workflow = StateGraph(state_class, checkpointer=checkpointer)

    # Add nodes
    for node_name, node_func in nodes.items():
        workflow.add_node(node_name, node_func)

    # Add edges
    for edge in edges:
        if len(edge) == 2:
            workflow.add_edge(edge[0], edge[1])
        elif len(edge) == 3:
            workflow.add_conditional_edges(edge[0], edge[1], edge[2])

    workflow.set_entry_point(entry_point)
    workflow.set_finish_point(END)

    return workflow.compile()
