from typing_extensions import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer
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
    format_recommended_steps,
    get_generic_summary,
)
from data_science_team_agent.tools.dataframe import get_dataframe_summary
from data_science_team_agent.utils.logging import log_ai_function, log_ai_error
from data_science_team_agent.utils.sandbox import run_code_sandboxed_subprocess
from data_science_team_agent.utils.messages import get_last_user_message_content
from data_science_team_agent.tools.sql import execute_sql_query

AGENT_NAME = "sql_database_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

class SQLDatabaseAgent(BaseAgent):
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
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
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

def make_sql_database_agent(
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
    llm = model

    if human_in_the_loop:
        if checkpointer is None:
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
        connection_string = state.get("connection_string")

        if not sql_query:
            return {"sql_error": "No SQL query generated"}

        try:
            message, result = execute_sql_query(
                query=sql_query,
                connection_string=connection_string
            )

            if "error" in result:
                return {"sql_error": result["error"]}
            else:
                return {"query_result": result.get("data", {}), "sql_query": sql_query}

        except Exception as e:
            return {"sql_error": f"SQL execution error: {str(e)}"}

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
