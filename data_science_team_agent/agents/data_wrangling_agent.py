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

AGENT_NAME = "data_wrangling_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

class DataWranglingAgent(BaseAgent):
    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_wrangler.py",
        function_name="data_wrangler",
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
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        self.response = self.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        return None

    def _make_compiled_graph(self):
        self.response = None
        return make_data_wrangling_agent(**self._params)

def make_data_wrangling_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_wrangler.py",
    function_name="data_wrangler",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    llm = model
    MAX_SUMMARY_COLUMNS = 30

    DEFAULT_WRANGLING_STEPS = format_recommended_steps(
        """
1. Analyze user instructions to understand data transformation requirements.
2. Apply filtering operations based on specified conditions.
3. Perform data aggregation or grouping as needed.
4. Create new columns through feature engineering.
5. Apply sorting operations for data organization.
6. Handle data merging or joining operations if required.
        """,
        heading="# Recommended Data Wrangling Steps:",
    )

    def _summarize_df_for_prompt(df: pd.DataFrame) -> str:
        df_limited = df.iloc[:, :MAX_SUMMARY_COLUMNS] if df.shape[1] > MAX_SUMMARY_COLUMNS else df
        summary = "\n\n".join(
            get_dataframe_summary(
                [df_limited],
                n_sample=min(n_samples, 5),
                skip_stats=True,
            )
        )
        MAX_CHARS = 5000
        return summary[:MAX_CHARS]

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
        recommended_steps: str
        data_raw: dict
        data_processed: dict
        all_datasets_summary: str
        data_wrangler_function: str
        data_wrangler_function_path: str
        data_wrangler_file_name: str
        data_wrangler_function_name: str
        data_wrangler_error: str
        max_retries: int
        retry_count: int

    def recommend_wrangling_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND WRANGLING STEPS")

        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Wrangling Expert. Given the following information about data and user instructions,
            recommend a series of steps to transform and manipulate the data.
            
            General Wrangling Operations:
            * Filtering data based on conditions
            * Sorting data for analysis
            * Grouping and aggregation operations
            * Feature engineering and column creation
            * Data merging and joining operations
            * Pivoting and reshaping data
            
            Custom Steps:
            * Analyze user instructions to understand specific transformation needs
            * Recommend operations that achieve the user's goals
            * Consider data types and structure for appropriate operations
            
            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list for effective data wrangling.
            """,
            input_variables=[
                "user_instructions",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary_str = _summarize_df_for_prompt(df)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Data Wrangling Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def create_wrangler_code(state: GraphState):
        print("    * CREATE DATA WRANGLER CODE")

        if bypass_recommended_steps:
            all_datasets_summary_str = _summarize_df_for_prompt(pd.DataFrame(state.get("data_raw")))
            steps_for_prompt = DEFAULT_WRANGLING_STEPS
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            steps_for_prompt = state.get("recommended_steps") or DEFAULT_WRANGLING_STEPS

        data_wrangling_prompt = PromptTemplate(
            template="""
            You are a Data Wrangling Agent. Your job is to create a {function_name}() function that transforms data according to user requirements.

            Recommended Steps:
            {recommended_steps}

            User Instructions:
            {user_instructions}

            Data Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that:
            1. Includes all imports inside the function
            2. Applies data transformation operations as specified
            3. Returns the transformed data as a DataFrame
            4. Handles edge cases and errors appropriately

            Function signature:
            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_processed

            Focus on creating reusable and efficient data transformation code.
            """,
            input_variables=[
                "recommended_steps",
                "user_instructions",
                "all_datasets_summary",
                "function_name",
            ],
        )

        data_wrangling_agent = data_wrangling_prompt | llm | PythonOutputParser()

        response = data_wrangling_agent.invoke(
            {
                "recommended_steps": steps_for_prompt,
                "user_instructions": state.get("user_instructions"),
                "all_datasets_summary": all_datasets_summary_str,
                "function_name": function_name,
            }
        )

        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "data_wrangler_function": response,
            "data_wrangler_function_path": file_path,
            "data_wrangler_file_name": file_name_2,
            "data_wrangler_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
            "recommended_steps": steps_for_prompt,
        }

    def execute_wrangler_code(state: GraphState):
        print("    * EXECUTE DATA WRANGLER CODE (SANDBOXED)")

        result, error = run_code_sandboxed_subprocess(
            code_snippet=state.get("data_wrangler_function"),
            function_name=state.get("data_wrangler_function_name"),
            data=state.get("data_raw"),
            timeout=10,
            memory_limit_mb=512,
        )

        if error is None:
            try:
                data_processed = result if isinstance(result, dict) else {}
                return {"data_processed": data_processed}
            except Exception as e:
                return {"data_processed": {}, "data_wrangler_error": f"Error processing wrangled data: {str(e)}"}
        else:
            return {"data_processed": {}, "data_wrangler_error": f"Wrangling execution error: {error}"}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("recommend_wrangling_steps", recommend_wrangling_steps)
    workflow.add_node("create_wrangler_code", create_wrangler_code)
    workflow.add_node("execute_wrangler_code", execute_wrangler_code)
    workflow.add_node("report_agent_outputs", lambda state: node_func_report_agent_outputs(state, [
        "recommended_steps",
        "data_wrangler_function",
        "data_wrangler_function_path",
        "data_wrangler_function_name",
        "data_wrangler_error",
        "data_processed",
    ]))
    
    workflow.add_edge(START, "recommend_wrangling_steps")
    workflow.add_edge("recommend_wrangling_steps", "create_wrangler_code")
    workflow.add_edge("create_wrangler_code", "execute_wrangler_code")
    workflow.add_edge("execute_wrangler_code", "report_agent_outputs")
    workflow.add_edge("report_agent_outputs", END)
    
    return workflow.compile()
