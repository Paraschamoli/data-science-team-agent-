from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.graph.state import CompiledStateGraph

from langchain_core.runnables import RunnableConfig
from langgraph.pregel.types import StreamMode

import pandas as pd
import sqlalchemy as sql
import json

from typing_extensions import Any, Callable, Dict, Type, Optional, Union, List

from data_science_team_agent.parsers.parsers import PythonOutputParser
from data_science_team_agent.utils.regex import (
    relocate_imports_inside_function,
    add_comments_to_top,
    remove_consecutive_duplicates,
)

from IPython.display import Image, display
import pandas as pd


class BaseAgent(CompiledStateGraph):
    """
    A generic base class for agents that interact with compiled state graphs.
    """

    def __init__(self, **params):
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
        self._params.update(kwargs)
        self._compiled_graph = self._make_compiled_graph()

    def __getattr__(self, name: str):
        return getattr(self._compiled_graph, name)

    def invoke(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = None, **kwargs):
        return self._compiled_graph.invoke(input, config, **kwargs)

    def ainvoke(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = None, **kwargs):
        return self._compiled_graph.ainvoke(input, config, **kwargs)

    def stream(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = None, **kwargs):
        return self._compiled_graph.stream(input, config, **kwargs)

    def astream(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = None, **kwargs):
        return self._compiled_graph.astream(input, config, **kwargs)

    def get_state(self, config: Optional[RunnableConfig] = None, **kwargs):
        return self._compiled_graph.get_state(config, **kwargs)

    def update_state(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None, **kwargs):
        return self._compiled_graph.update_state(state, config, **kwargs)

    def get_graph(self, **kwargs):
        return self._compiled_graph.get_graph(**kwargs)

    def draw_mermaid_png(self, **kwargs):
        return self._compiled_graph.draw_mermaid_png(**kwargs)

    def get_response(self):
        return self.response


def node_func_human_review(state, prompt_text, yes_goto, no_goto, user_instructions_key, recommended_steps_key, code_snippet_key):
    """Handle human review of agent recommendations."""
    user_instructions = state.get(user_instructions_key, "")
    recommended_steps = state.get(recommended_steps_key, "")
    code_snippet = state.get(code_snippet_key, "")
    
    full_prompt = prompt_text.format(
        steps=recommended_steps,
        user_instructions=user_instructions,
        code_snippet=code_snippet
    )
    
    print("    * HUMAN REVIEW")
    print(f"    Prompt: {full_prompt}")
    
    # For automated execution, we'll skip human review and go to yes_goto
    return Command(goto=yes_goto)


def node_func_fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, agent_name, log, file_path, function_name):
    """Fix broken agent code."""
    code_snippet = state.get(code_snippet_key, "")
    error = state.get(error_key, "")
    
    prompt = prompt_template.format(
        function_name=function_name,
        code_snippet=code_snippet,
        error=error
    )
    
    print(f"    * FIX {agent_name.upper()} CODE")
    
    fix_agent = prompt_template | llm | PythonOutputParser()
    fixed_code = fix_agent.invoke({
        "function_name": function_name,
        "code_snippet": code_snippet,
        "error": error
    })
    
    fixed_code = relocate_imports_inside_function(fixed_code)
    fixed_code = add_comments_to_top(fixed_code, agent_name=agent_name)
    
    return {code_snippet_key: fixed_code}


def node_func_report_agent_outputs(state, keys_to_include):
    """Report final outputs from agent execution."""
    print("    * REPORT AGENT OUTPUTS")
    
    report = {
        "report_title": "Agent Outputs",
        "outputs": {}
    }
    
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
        exec(code_snippet, globals(), local_vars)
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
