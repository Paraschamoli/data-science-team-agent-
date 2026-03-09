from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict, Annotated, Sequence, Union
import pandas as pd
import json
from data_science_team_agent.templates import BaseAgent
from data_science_team_agent.agents import DataWranglingAgent, DataVisualizationAgent
from data_science_team_agent.utils.plotly import plotly_from_dict

AGENT_NAME = "pandas_data_analyst"

class PandasDataAnalyst(BaseAgent):
    def __init__(self, model, data_wrangling_agent, data_visualization_agent, checkpointer=None):
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
        response = self._compiled_graph.invoke({
            "messages": [("user", user_instructions)],
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None

    def get_data_wrangled(self):
        if self.response:
            return pd.DataFrame(self.response.get("data_wrangled"))
        return None

    def get_plotly_graph(self):
        if self.response:
            plot_dict = self.response.get("plotly_graph")
            if plot_dict:
                return plotly_from_dict(plot_dict)
        return None

def make_pandas_data_analyst(model, data_wrangling_agent, data_visualization_agent, checkpointer=None):
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
            user_instructions=state["user_instructions"],
            data_raw=pd.DataFrame(state["data_raw"])
        )
        response = data_wrangling_agent.get_response()
        return {"data_wrangled": response.get("data_processed", state["data_raw"])}

    def create_visualization(state: GraphState):
        print("    * CREATE VISUALIZATION")
        data_visualization_agent.invoke_agent(
            user_instructions=state["user_instructions"],
            data_raw=pd.DataFrame(state["data_wrangled"])
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
