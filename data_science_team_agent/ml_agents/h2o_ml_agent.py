"""H2O AutoML agent for automated machine learning."""

import operator
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage  # type: ignore[import]
from langgraph.graph import END, START  # type: ignore[import]
from typing_extensions import TypedDict

from data_science_team_agent.templates import (
    BaseAgent,
    create_coding_agent_graph,
)
from data_science_team_agent.tools.h2o import (
    get_h2o_model_summary,
    initialize_h2o,
    train_h2o_model,
)
from data_science_team_agent.utils.regex import (
    format_agent_name,
)

AGENT_NAME = "h2o_ml_agent"


class H2OMLAgent(BaseAgent):
    """Agent responsible for automated machine learning using H2O AutoML."""

    def __init__(self, model, checkpointer=None):
        """Initialize the H2O ML agent.

        Args:
            model: The language model to use.
            checkpointer: Optional checkpointer for state management.

        """
        self._params = {
            "model": model,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_h2o_ml_agent(**self._params)

    def invoke_agent(
        self, user_instructions=None, data_raw=None, target_variable=None, max_retries=3, retry_count=0, **kwargs
    ):
        """Execute the H2O ML agent workflow.

        Args:
            user_instructions: Optional user instructions.
            data_raw: Optional raw data to process.
            target_variable: Optional target variable for supervised learning.
            max_retries: Maximum number of retries.
            retry_count: Current retry count.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.

        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None


def make_h2o_ml_agent(model, checkpointer=None):
    """Create an H2O AutoML agent for automated machine learning.

    Args:
        model: The language model to use.
        checkpointer: Optional checkpointer for state management.

    Returns:
        Compiled H2O ML agent.

    """

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
        _, model_info = train_h2o_model(data=data_raw, target_column=target_var)

        if "error" in model_info:
            return {"h2o_model": {}, "error": model_info["error"]}

        return {"h2o_model": model_info, "model_performance": model_info.get("performance", {})}

    def get_model_summary(state: GraphState):
        print("    * GET MODEL SUMMARY")

        model_info = state.get("h2o_model", {})
        model_id = model_info.get("model_id")

        if not model_id:
            return {"model_summary": "No model available for summary"}

        _, summary = get_h2o_model_summary(model_id)

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
        nodes=nodes, edges=edges, state_class=GraphState, entry_point="initialize_h2o", checkpointer=checkpointer
    )
