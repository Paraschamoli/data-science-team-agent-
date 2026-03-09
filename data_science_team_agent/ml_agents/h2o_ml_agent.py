from typing_extensions import TypedDict, Annotated, Sequence
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
import os
import json
import pandas as pd
from data_science_team_agent.templates import (
    node_func_human_review,
    node_func_fix_agent_code,
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from data_science_team_agent.parsers.parsers import PythonOutputParser
from data_science_team_agent.utils.regex import (
    relocate_imports_inside_function,
    add_comments_to_top,
    format_agent_name,
    get_generic_summary,
)
from data_science_team_agent.tools.h2o import (
    train_h2o_model,
    predict_with_h2o_model,
    get_h2o_model_summary,
    initialize_h2o,
)

AGENT_NAME = "h2o_ml_agent"

class H2OMLAgent(BaseAgent):
    def __init__(self, model, checkpointer=None):
        self._params = {
            "model": model,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_h2o_ml_agent(**self._params)

    def invoke_agent(self, user_instructions=None, data_raw=None, target_variable=None, max_retries=3, retry_count=0, **kwargs):
        response = self._compiled_graph.invoke({
            "messages": [("user", user_instructions)] if user_instructions else [],
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict() if data_raw is not None else None,
            "target_variable": target_variable,
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None

def make_h2o_ml_agent(model, checkpointer=None):
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        target_variable: str
        h2o_model: dict
        model_performance: dict
        max_retries: int
        retry_count: int

    def initialize_h2o_cluster(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * INITIALIZE H2O CLUSTER")

        init_result = initialize_h2o()
        return {"h2o_status": init_result}

    def train_model(state: GraphState):
        print("    * TRAIN H2O MODEL")

        data_raw = state.get("data_raw")
        target_var = state.get("target_variable")
        
        if not data_raw:
            return {"h2o_model": {}, "error": "No data provided for training"}
        
        if not target_var:
            return {"h2o_model": {}, "error": "Target variable not specified"}
        
        # Train H2O model
        message, model_info = train_h2o_model(
            data=data_raw,
            target_column=target_var
        )
        
        if "error" in model_info:
            return {"h2o_model": {}, "error": model_info["error"]}
        
        return {
            "h2o_model": model_info,
            "model_performance": model_info.get("performance", {})
        }

    def get_model_summary(state: GraphState):
        print("    * GET MODEL SUMMARY")

        model_info = state.get("h2o_model", {})
        model_id = model_info.get("model_id")
        
        if not model_id:
            return {"model_summary": "No model available for summary"}
        
        message, summary = get_h2o_model_summary(model_id)
        
        if "error" in summary:
            return {"model_summary": summary["error"]}
        
        return {"model_summary": summary}

    def report_outputs(state: GraphState):
        print("    * REPORT ML OUTPUTS")
        
        report = {
            "report_title": "H2O Machine Learning Results",
            "h2o_model": state.get("h2o_model", {}),
            "model_performance": state.get("model_performance", {}),
            "model_summary": state.get("model_summary", ""),
            "target_variable": state.get("target_variable", ""),
        }
        
        return {"agent_outputs": report}

    # Create workflow
    nodes = {
        "initialize_h2o": initialize_h2o_cluster,
        "train_model": train_model,
        "get_model_summary": get_model_summary,
        "report_outputs": report_outputs,
    }
    
    edges = [
        (START, "initialize_h2o"),
        ("initialize_h2o", "train_model"),
        ("train_model", "get_model_summary"),
        ("get_model_summary", "report_outputs"),
        ("report_outputs", END),
    ]
    
    return create_coding_agent_graph(
        nodes=nodes,
        edges=edges,
        state_class=GraphState,
        entry_point="initialize_h2o",
        checkpointer=checkpointer
    )
