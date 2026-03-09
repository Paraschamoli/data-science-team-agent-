"""Pandas data analyst for DataFrame-based data analysis.

Provides specialized multi-agent implementation for analyzing
pandas DataFrames with automated data wrangling, visualization,
and statistical analysis capabilities.
"""

from collections.abc import Sequence
from typing import Annotated

import pandas as pd
from langchain_core.messages import BaseMessage  # type: ignore[import]
from langgraph.graph import END, START, StateGraph  # type: ignore[import]
from typing_extensions import TypedDict

from data_science_team_agent.templates import BaseAgent
from data_science_team_agent.utils.plotly import plotly_from_dict

AGENT_NAME = "pandas_data_analyst"


class PandasDataAnalyst(BaseAgent):
    """Multi-agent for pandas DataFrame analysis and visualization."""

    def __init__(self, model, data_wrangling_agent, data_visualization_agent, checkpointer=None):
        """Initialize the pandas data analyst.

        Args:
            model: The language model to use.
            data_wrangling_agent: The data wrangling agent to use.
            data_visualization_agent: The data visualization agent to use.
            checkpointer: Checkpointer for state management. Defaults to None.
        """
        self._params = {
            "model": model,
            "data_wrangling_agent": data_wrangling_agent,
            "data_visualization_agent": data_visualization_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_pandas_data_analyst(**self._params)

    def invoke_agent(self, user_instructions, data_raw, max_retries=3, retry_count=0, **kwargs):
        """Execute the agent workflow.

        Args:
            user_instructions: User instructions for analysis.
            data_raw: Raw pandas DataFrame to analyze.
            max_retries: Maximum number of retries. Defaults to 3.
            retry_count: Current retry count. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.
        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_data_wrangled(self):
        """Get the wrangled data from the last execution."""
        if self.response:
            return pd.DataFrame(self.response.get("data_wrangled"))
        return None

    def get_plotly_graph(self):
        """Get the Plotly graph from the last execution."""
        if self.response:
            plot_dict = self.response.get("plotly_graph")
            if plot_dict:
                return plotly_from_dict(plot_dict)
        return None


def make_pandas_data_analyst(model, data_wrangling_agent, data_visualization_agent, checkpointer=None):
    """Create a pandas data analyst multi-agent.

    Args:
        model: The language model to use.
        data_wrangling_agent: The data wrangling agent to use.
        data_visualization_agent: The data visualization agent to use.
        checkpointer: Checkpointer for state management. Defaults to None.

    Returns:
        Compiled pandas data analyst agent graph.
    """

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], "add_messages"]
        user_instructions: str
        data_raw: dict
        data_wrangled: dict
        plotly_graph: dict
        max_retries: int
        retry_count: int

    def wrangle_data(state: GraphState):
        print("    * WRANGLE DATA")
        data_wrangling_agent.invoke_agent(
            user_instructions=state["user_instructions"], data_raw=pd.DataFrame(state["data_raw"])
        )
        response = data_wrangling_agent.get_response()
        return {"data_wrangled": response.get("data_processed", state["data_raw"])}

    def create_visualization(state: GraphState):
        print("    * CREATE VISUALIZATION")
        data_visualization_agent.invoke_agent(
            user_instructions=state["user_instructions"], data_raw=pd.DataFrame(state["data_wrangled"])
        )
        response = data_visualization_agent.get_response()
        return {"plotly_graph": response.get("plot_data", {})}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("wrangle_data", wrangle_data)
    workflow.add_node("create_visualization", create_visualization)
    workflow.add_edge(START, "wrangle_data")
    workflow.add_edge("wrangle_data", "create_visualization")
    workflow.add_edge("create_visualization", END)

    return workflow.compile()
