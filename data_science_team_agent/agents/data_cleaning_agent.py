from typing_extensions import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
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
    format_recommended_steps,
    get_generic_summary,
)
from data_science_team_agent.tools.dataframe import get_dataframe_summary
from data_science_team_agent.utils.logging import log_ai_function, log_ai_error
from data_science_team_agent.utils.sandbox import run_code_sandboxed_subprocess
from data_science_team_agent.utils.messages import get_last_user_message_content

AGENT_NAME = "data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
MAX_SUMMARY_COLUMNS = 30

class DataCleaningAgent(BaseAgent):
    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_cleaner.py",
        function_name="data_cleaner",
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
        return make_data_cleaning_agent(**self._params)

def make_data_cleaning_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    llm = model
    MAX_SUMMARY_COLUMNS = 30

    DEFAULT_CLEANING_STEPS = format_recommended_steps(
        """
1. Remove columns with >40% missing values.
2. Impute numeric missing values with the mean; impute categorical missing with the mode.
3. Convert columns to appropriate data types (numeric/categorical/datetime).
4. Remove duplicate rows.
5. Optionally drop rows with remaining missing values if still present.
6. Remove extreme outliers (values beyond 3x IQR) for numeric columns unless instructed otherwise.
        """,
        heading="# Recommended Data Cleaning Steps:",
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
            print("Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver().")
            checkpointer = MemorySaver()

    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")

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
        data_cleaned: dict
        all_datasets_summary: str
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_file_name: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        data_cleaning_summary: str
        data_cleaner_error_log_path: str
        max_retries: int
        retry_count: int

    def recommend_cleaning_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND CLEANING STEPS")

        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Expert. Given the following information about data, 
            recommend a series of numbered steps to take to clean and preprocess it. 
            The steps should be tailored to data characteristics and should be helpful 
            for a data cleaning agent that will be implemented.
            
            General Steps:
            Things that should be considered in data cleaning steps:
            
            * Removing columns if more than 40 percent of data is missing
            * Imputing missing values with mean of column if column is numeric
            * Imputing missing values with mode of column if column is categorical
            * Converting columns to correct data type
            * Removing duplicate rows
            * Removing rows with missing values
            * Removing rows with extreme outliers (3X interquartile range)
            
            Custom Steps:
            * Analyze the data to determine if any additional data cleaning steps are needed.
            * Recommend steps that are specific to data provided. Include why these steps are necessary or beneficial.
            * If no additional steps are needed, simply state that no additional steps are required.
            
            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
            
            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to data cleaning.
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
                heading="# Recommended Data Cleaning Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def create_data_cleaner_code(state: GraphState):
        print("    * CREATE DATA CLEANER CODE")

        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))
            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)
            all_datasets_summary_str = _summarize_df_for_prompt(df)
            steps_for_prompt = DEFAULT_CLEANING_STEPS
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            steps_for_prompt = state.get("recommended_steps") or DEFAULT_CLEANING_STEPS

        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Your job is to create a {function_name}() function that can be run on data provided using the following recommended steps.

            Recommended Steps:
            {recommended_steps}

            You can use Pandas, Numpy, and Scikit Learn libraries to clean data.

            Below are summaries of all datasets provided. Use this information about data to help determine how to clean data:

            {all_datasets_summary}

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.

            Return code to provide data cleaning function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_cleaned

            Best Practices and Error Preventions:

            Always ensure that when assigning the output of fit_transform() from SimpleImputer to a Pandas DataFrame column, you call .ravel() or flatten the array, because fit_transform() returns a 2D array while a DataFrame column is 1D.
            - Do NOT hardcode column names; derive columns programmatically from provided data and user instructions.
            
            """,
            input_variables=[
                "recommended_steps",
                "all_datasets_summary",
                "function_name",
            ],
        )

        data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()

        response = data_cleaning_agent.invoke(
            {
                "recommended_steps": steps_for_prompt,
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
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_file_name": file_name_2,
            "data_cleaner_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
            "recommended_steps": steps_for_prompt,
        }

    def execute_data_cleaner_code(state: GraphState):
        print("    * EXECUTE DATA CLEANER CODE (SANDBOXED)")

        result, error = run_code_sandboxed_subprocess(
            code_snippet=state.get("data_cleaner_function"),
            function_name=state.get("data_cleaner_function_name"),
            data=state.get("data_raw"),
            timeout=10,
            memory_limit_mb=512,
        )

        data_cleaning_summary = None
        df_out = None
        validation_error = None
        if error is None:
            try:
                df_out = pd.DataFrame(result)
                df_raw = pd.DataFrame(state.get("data_raw"))

                rows_before, rows_after = len(df_raw), len(df_out)
                cols_before, cols_after = set(df_raw.columns), set(df_out.columns)
                dropped_cols = sorted(list(cols_before - cols_after))
                added_cols = sorted(list(cols_after - cols_before))
                dtype_changes = []
                for col in df_raw.columns:
                    if col in df_out.columns:
                        before = str(df_raw[col].dtype)
                        after = str(df_out[col].dtype)
                        if before != after:
                            dtype_changes.append(f"{col}: {before} -> {after}")

                data_cleaning_summary = "\n".join(
                    [
                        "# Data Cleaning Summary",
                        f"Rows: {rows_before} -> {rows_after} (Δ {rows_after - rows_before})",
                        f"Columns: {len(cols_before)} -> {len(cols_after)} (Δ {len(cols_after) - len(cols_before)})",
                        f"Dropped Columns: {', '.join(dropped_cols) if dropped_cols else 'None'}",
                        f"Added Columns: {', '.join(added_cols) if added_cols else 'None'}",
                        "Dtype Changes:",
                        "\n".join(dtype_changes) if dtype_changes else "None",
                    ]
                )
            except Exception as exc:
                validation_error = f"Cleaned output is not a valid table: {exc}"
        else:
            validation_error = error

        error_prefixed = (
            f"An error occurred during data cleaning: {validation_error}"
            if validation_error
            else None
        )

        error_log_path = None
        if error_prefixed and log:
            error_log_path = log_ai_error(
                error_message=error_prefixed,
                file_name=f"{file_name}_errors.log",
                log=log,
                log_path=log_path if log_path is not None else LOG_PATH,
                overwrite=False,
            )
            if error_log_path:
                print(f"      Error logged to: {error_log_path}")

        return {
            "data_cleaned": df_out.to_dict() if error_prefixed is None else None,
            "data_cleaner_error": error_prefixed,
            "data_cleaning_summary": data_cleaning_summary,
            "data_cleaner_error_log_path": error_log_path,
        }

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("recommend_cleaning_steps", recommend_cleaning_steps)
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("report_agent_outputs", lambda state: node_func_report_agent_outputs(state, [
        "recommended_steps",
        "data_cleaner_function",
        "data_cleaner_function_path",
        "data_cleaner_function_name",
        "data_cleaner_error",
        "data_cleaning_summary",
        "data_cleaner_error_log_path",
    ]))
    
    workflow.add_edge(START, "recommend_cleaning_steps")
    workflow.add_edge("recommend_cleaning_steps", "create_data_cleaner_code")
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("execute_data_cleaner_code", "report_agent_outputs")
    workflow.add_edge("report_agent_outputs", END)
    
    return workflow.compile()
