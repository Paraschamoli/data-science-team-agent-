"""SQL data analyst for database-driven data analysis.

Provides specialized multi-agent implementation for analyzing data
from SQL databases with automated query generation and result
interpretation.
"""

from collections.abc import Sequence
from typing import Annotated

import pandas as pd
from langchain_core.messages import BaseMessage  # type: ignore[import]
from langgraph.graph import END, START, StateGraph  # type: ignore[import]
from typing_extensions import TypedDict

from data_science_team_agent.templates import BaseAgent

AGENT_NAME = "sql_data_analyst"


class SQLDataAnalyst(BaseAgent):
    """Multi-agent for SQL data analysis and querying."""

    def __init__(self, model, sql_database_agent, checkpointer=None):
        """Initialize the SQL data analyst.

        Args:
            model: The language model to use.
            sql_database_agent: The SQL database agent to use.
            checkpointer: Checkpointer for state management. Defaults to None.
        """
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
        """Execute the agent workflow.

        Args:
            user_instructions: User instructions for analysis.
            connection_string: Database connection string.
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
                "connection_string": connection_string,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_query_results(self):
        """Get the query results from the last execution."""
        if self.response:
            return pd.DataFrame(self.response.get("query_results", {}))
        return None


def make_sql_data_analyst(model, sql_database_agent, checkpointer=None):
    """Create a SQL data analyst multi-agent.

    Args:
        model: The language model to use.
        sql_database_agent: The SQL database agent to use.
        checkpointer: Checkpointer for state management. Defaults to None.

    Returns:
        Compiled SQL data analyst agent graph.
    """

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
            user_instructions=state["user_instructions"], connection_string=state["connection_string"]
        )
        response = sql_database_agent.get_response()
        return {"query_results": response.get("query_result", {})}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("generate_query", generate_query)
    workflow.add_edge(START, "generate_query")
    workflow.add_edge("generate_query", END)

    return workflow.compile()
