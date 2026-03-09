from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict, Annotated, Sequence, Union
import pandas as pd
import json
from data_science_team_agent.templates import BaseAgent
from data_science_team_agent.agents import SQLDatabaseAgent

AGENT_NAME = "sql_data_analyst"

class SQLDataAnalyst(BaseAgent):
    def __init__(self, model, sql_database_agent, checkpointer=None):
        self._params = {
            "model": model,
            "sql_database_agent": sql_database_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_sql_data_analyst(**self._params)

    def invoke_agent(self, user_instructions, connection_string, max_retries=3, retry_count=0, **kwargs):
        response = self._compiled_graph.invoke({
            "messages": [("user", user_instructions)],
            "user_instructions": user_instructions,
            "connection_string": connection_string,
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None

    def get_query_results(self):
        if self.response:
            return pd.DataFrame(self.response.get("query_results", {}))
        return None

def make_sql_data_analyst(model, sql_database_agent, checkpointer=None):
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], "add_messages"]
        user_instructions: str
        connection_string: str
        query_results: dict
        max_retries: int
        retry_count: int

    def generate_query(state: GraphState):
        print("    * GENERATE SQL QUERY")
        sql_database_agent.invoke_agent(
            user_instructions=state["user_instructions"],
            connection_string=state["connection_string"]
        )
        response = sql_database_agent.get_response()
        return {"query_results": response.get("query_result", {})}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("generate_query", generate_query)
    workflow.add_edge(START, "generate_query")
    workflow.add_edge("generate_query", END)
    
    return workflow.compile()
