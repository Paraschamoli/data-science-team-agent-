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
from data_science_team_agent.tools.mlflow import (
    create_mlflow_experiment,
    log_experiment_to_mlflow,
    get_mlflow_run_info,
    list_mlflow_experiments,
    log_model_to_mlflow,
)

AGENT_NAME = "mlflow_tools_agent"

class MLflowToolsAgent(BaseAgent):
    def __init__(self, model, checkpointer=None):
        self._params = {
            "model": model,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_mlflow_tools_agent(**self._params)

    def invoke_agent(self, user_instructions=None, max_retries=3, retry_count=0, **kwargs):
        response = self._compiled_graph.invoke({
            "messages": [("user", user_instructions)] if user_instructions else [],
            "user_instructions": user_instructions,
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None

def make_mlflow_tools_agent(model, checkpointer=None):
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        mlflow_action: str
        experiment_info: dict
        run_info: dict
        max_retries: int
        retry_count: int

    def parse_mlflow_action(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * PARSE MLFLOW ACTION")

        user_instructions = state.get("user_instructions", "").lower()
        
        if "create experiment" in user_instructions:
            action = "create_experiment"
        elif "log experiment" in user_instructions or "log run" in user_instructions:
            action = "log_experiment"
        elif "list experiments" in user_instructions:
            action = "list_experiments"
        elif "get run" in user_instructions or "run info" in user_instructions:
            action = "get_run_info"
        elif "log model" in user_instructions:
            action = "log_model"
        else:
            action = "help"
        
        return {"mlflow_action": action}

    def execute_mlflow_action(state: GraphState):
        print(f"    * EXECUTE {state['mlflow_action']}")

        action = state.get("mlflow_action")
        user_instructions = state.get("user_instructions", "")
        
        if action == "create_experiment":
            # Extract experiment name from instructions
            words = user_instructions.split()
            exp_name = "Default Experiment"
            for i, word in enumerate(words):
                if word.lower() == "experiment" and i + 1 < len(words):
                    exp_name = " ".join(words[i+1:i+3])  # Take next 1-2 words
                    break
            
            message, info = create_mlflow_experiment(exp_name)
            return {"experiment_info": info}
        
        elif action == "log_experiment":
            # Create dummy experiment data
            metrics = {"accuracy": 0.85, "loss": 0.15}
            parameters = {"model_type": "test", "epochs": 10}
            
            message, info = log_experiment_to_mlflow(
                model_data={},
                metrics=metrics,
                parameters=parameters,
                experiment_name="Test Experiment"
            )
            return {"run_info": info}
        
        elif action == "list_experiments":
            message, info = list_mlflow_experiments()
            return {"experiment_info": info}
        
        elif action == "get_run_info":
            message, info = get_mlflow_run_info()
            return {"run_info": info}
        
        elif action == "log_model":
            # This would require an actual model object
            message, info = log_model_to_mlflow(
                model_object={},
                model_name="test_model",
                model_type="sklearn"
            )
            return {"run_info": info}
        
        else:
            help_text = """
            Available MLflow actions:
            - Create experiment: "create experiment [name]"
            - Log experiment: "log experiment" or "log run"
            - List experiments: "list experiments"
            - Get run info: "get run info" or "run info"
            - Log model: "log model [name]"
            """
            return {"experiment_info": {"help": help_text}}

    def report_outputs(state: GraphState):
        print("    * REPORT MLFLOW OUTPUTS")
        
        report = {
            "report_title": "MLflow Tools Results",
            "mlflow_action": state.get("mlflow_action", ""),
            "experiment_info": state.get("experiment_info", {}),
            "run_info": state.get("run_info", {}),
        }
        
        return {"agent_outputs": report}

    # Create workflow
    nodes = {
        "parse_mlflow_action": parse_mlflow_action,
        "execute_mlflow_action": execute_mlflow_action,
        "report_outputs": report_outputs,
    }
    
    edges = [
        (START, "parse_mlflow_action"),
        ("parse_mlflow_action", "execute_mlflow_action"),
        ("execute_mlflow_action", "report_outputs"),
        ("report_outputs", END),
    ]
    
    return create_coding_agent_graph(
        nodes=nodes,
        edges=edges,
        state_class=GraphState,
        entry_point="parse_mlflow_action",
        checkpointer=checkpointer
    )
