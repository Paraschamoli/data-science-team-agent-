"""SQL database agent for automated database operations.

Provides specialized agent capabilities for connecting to and
querying SQL databases with automated workflows for data
extraction and analysis.
"""

import operator
import os
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage  # type: ignore[import]
from langchain_core.prompts import PromptTemplate  # type: ignore[import]
from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
from langgraph.graph import END, START, StateGraph  # type: ignore[import]
from langgraph.types import Checkpointer  # type: ignore[import]
from typing_extensions import TypedDict

from data_science_team_agent.templates import (
    BaseAgent,
)
from data_science_team_agent.utils.regex import (
    format_agent_name,
)

AGENT_NAME = "sql_database_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class SQLDatabaseAgent(BaseAgent):
    """Agent for SQL database querying and interaction."""

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="sql_agent.py",
        function_name="sql_query_generator",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer: Checkpointer = None,
    ):
        """Initialize the SQL database agent.

        Args:
            model: The language model to use.
            n_samples: Number of samples to process. Defaults to 30.
            log: Whether to enable logging. Defaults to False.
            log_path: Path to log file. Defaults to None.
            file_name: Name of the output file. Defaults to "sql_agent.py".
            function_name: Name of the function. Defaults to "sql_query_generator".
            overwrite: Whether to overwrite existing files. Defaults to True.
            human_in_the_loop: Whether to enable human-in-the-loop. Defaults to False.
            bypass_recommended_steps: Whether to bypass recommended steps. Defaults to False.
            bypass_explain_code: Whether to bypass code explanation. Defaults to False.
            checkpointer: Checkpointer for state management. Defaults to None.
        """
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def invoke_agent(
        self,
        connection_string: str,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Execute the agent workflow.

        Args:
            connection_string: Connection string.
            user_instructions: Optional user instructions. Defaults to None.
            max_retries: Maximum number of retries. Defaults to 3.
            retry_count: Current retry count. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.

        """
        self.response = self.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "connection_string": connection_string,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        return None

    def _make_compiled_graph(self):
        self.response = None
        return make_sql_database_agent(**self._params)


def make_sql_database_agent(  # noqa: C901 - complex agent setup is intentional
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="sql_agent.py",
    function_name="sql_query_generator",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    """Create a SQL database agent.

    Args:
        model: The language model to use.
        n_samples: Number of samples to process. Defaults to 30.
        log: Whether to enable logging. Defaults to False.
        log_path: Path to log file. Defaults to None.
        file_name: Name of the output file. Defaults to "sql_agent.py".
        function_name: Name of the function. Defaults to "sql_query_generator".
        overwrite: Whether to overwrite existing files. Defaults to True.
        human_in_the_loop: Whether to enable human-in-the-loop. Defaults to False.
        bypass_recommended_steps: Whether to bypass recommended steps. Defaults to False.
        bypass_explain_code: Whether to bypass code explanation. Defaults to False.
        checkpointer: Checkpointer for state management. Defaults to None.

    Returns:
        Compiled SQL database agent graph.
    """
    llm = model

    if human_in_the_loop and checkpointer is None:
        checkpointer = MemorySaver()

    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        connection_string: str
        recommended_steps: str
        query_result: dict
        sql_query: str
        sql_function: str
        sql_function_path: str
        sql_file_name: str
        sql_function_name: str
        sql_error: str
        max_retries: int
        retry_count: int

    def generate_sql_query(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * GENERATE SQL QUERY")

        sql_prompt = PromptTemplate(
            template="""
            You are a SQL Expert. Given the following user instructions and database context,
            generate an appropriate SQL query to fulfill the request.

            SQL Best Practices:
            * Use proper JOIN syntax when combining tables
            * Apply appropriate WHERE clauses for filtering
            * Use GROUP BY and HAVING for aggregation
            * Apply proper ORDER BY for sorting
            * Use LIMIT to restrict results when appropriate
            * Handle NULL values appropriately

            User instructions:
            {user_instructions}

            Connection String: {connection_string}

            Return SQL query in a single code block:
            ```sql
            SELECT ...
            ```

            Focus on writing efficient, readable SQL queries that accomplish the user's goals.
            """,
            input_variables=[
                "user_instructions",
                "connection_string",
            ],
        )

        sql_agent = sql_prompt | llm

        response = sql_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "connection_string": state.get("connection_string"),
        })

        # Extract SQL from response
        sql_query = response.content.strip()
        if "```sql" in sql_query:
            start = sql_query.find("```sql") + 6
            end = sql_query.find("```", start)
            if end != -1:
                sql_query = sql_query[start:end].strip()

        return {"sql_query": sql_query}

    def execute_sql_query(state: GraphState):
        print("    * EXECUTE SQL QUERY")

        sql_query = state.get("sql_query")

        if not sql_query:
            return {"sql_error": "No SQL query generated"}

        try:
            _, result = execute_sql_query(state)

            if "error" in result:
                return {"sql_error": result["error"]}
            else:
                return {"query_result": result.get("data", {}), "sql_query": sql_query}

        except Exception as e:
            return {"sql_error": f"SQL execution error: {e!s}"}

    def report_outputs(state: GraphState):
        print("    * REPORT SQL OUTPUTS")

        report = {
            "report_title": "SQL Database Agent Results",
            "sql_query": state.get("sql_query", ""),
            "query_result": state.get("query_result", {}),
            "sql_error": state.get("sql_error", ""),
        }

        return {"agent_outputs": report}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("generate_sql_query", generate_sql_query)
    workflow.add_node("execute_sql_query", execute_sql_query)
    workflow.add_node("report_outputs", report_outputs)

    workflow.add_edge(START, "generate_sql_query")
    workflow.add_edge("generate_sql_query", "execute_sql_query")
    workflow.add_edge("execute_sql_query", "report_outputs")
    workflow.add_edge("report_outputs", END)

    return workflow.compile()
