"""Feature engineering agent for creating new features from existing data."""

import operator
import os
from collections.abc import Sequence
from typing import Annotated

import pandas as pd
from langchain_core.messages import BaseMessage  # type: ignore[import]
from langchain_core.prompts import PromptTemplate  # type: ignore[import]
from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
from langgraph.graph import END, START, StateGraph  # type: ignore[import]
from langgraph.types import Checkpointer  # type: ignore[import]
from typing_extensions import TypedDict

from data_science_team_agent.parsers.parsers import PythonOutputParser
from data_science_team_agent.templates import (
    BaseAgent,
    node_func_report_agent_outputs,
)
from data_science_team_agent.tools.dataframe import get_dataframe_summary
from data_science_team_agent.utils.logging import log_ai_function
from data_science_team_agent.utils.regex import (
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    relocate_imports_inside_function,
)
from data_science_team_agent.utils.sandbox import run_code_sandboxed_subprocess

AGENT_NAME = "feature_engineering_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class FeatureEngineeringAgent(BaseAgent):
    """Agent responsible for creating new features from existing data."""

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="feature_engineer.py",
        function_name="feature_engineer",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        **kwargs,
    ):
        """Initialize the feature engineering agent.

        Args:
            model: The language model to use.
            n_samples: Number of samples to generate.
            log: Whether to log output.
            log_path: Path to log file.
            file_name: Name of the generated file.
            function_name: Name of the generated function.
            overwrite: Whether to overwrite existing files.
            human_in_the_loop: Whether to enable human-in-the-loop.
            bypass_recommended_steps: Whether to bypass recommended steps.
            bypass_explain_code: Whether to bypass code explanation.
            **kwargs: Additional keyword arguments.

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
            "checkpointer": None,  # Add checkpointer to params
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        target_variable: str | None = None,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Execute the feature engineering agent workflow.

        Args:
            data_raw: The raw data to process.
            target_variable: Optional target variable for supervised learning.
            user_instructions: Optional user instructions.
            max_retries: Maximum number of retries.
            retry_count: Current retry count.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.

        """
        self.response = self.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        return None

    def _make_compiled_graph(self):
        self.response = None
        return make_feature_engineering_agent(**self._params)


def make_feature_engineering_agent(  # noqa: C901 - complex agent setup is intentional
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="feature_engineer.py",
    function_name="feature_engineer",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    """Create a feature engineering agent for generating new features.

    Args:
        model: The language model to use.
        n_samples: Number of samples to generate.
        log: Whether to log output.
        log_path: Path to log file.
        file_name: Name of the generated file.
        function_name: Name of the generated function.
        overwrite: Whether to overwrite existing files.
        human_in_the_loop: Whether to enable human-in-the-loop.
        bypass_recommended_steps: Whether to bypass recommended steps.
        bypass_explain_code: Whether to bypass code explanation.
        checkpointer: Checkpointer to use.

    Returns:
        Compiled feature engineering agent.

    """
    llm = model
    MAX_SUMMARY_COLUMNS = 30

    DEFAULT_FEATURE_STEPS = format_recommended_steps(
        """
1. Analyze existing columns and data types.
2. Create interaction features between variables.
3. Generate polynomial features for numeric variables.
4. Create categorical encoding features.
5. Extract datetime features if date columns exist.
6. Create aggregate features (mean, sum, count) by groups.
7. Apply scaling or normalization to numeric features.
        """,
        heading="# Recommended Feature Engineering Steps:",
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
        target_variable: str
        recommended_steps: str
        data_raw: dict
        data_featured: dict
        all_datasets_summary: str
        feature_engineer_function: str
        feature_engineer_function_path: str
        feature_engineer_file_name: str
        feature_engineer_function_name: str
        feature_engineer_error: str
        max_retries: int
        retry_count: int

    def recommend_feature_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND FEATURE ENGINEERING STEPS")

        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Expert. Given the following information about data and user instructions,
            recommend a series of steps to create meaningful features for machine learning.

            General Feature Engineering Operations:
            * Create interaction features between variables
            * Generate polynomial features for numeric variables
            * Apply categorical encoding techniques
            * Extract datetime features from date columns
            * Create aggregate features by grouping
            * Apply scaling and normalization
            * Handle text features if present

            Custom Steps:
            * Analyze target variable if provided for supervised feature creation
            * Consider domain-specific feature creation
            * Recommend features that improve model performance

            User instructions:
            {user_instructions}

            Target Variable:
            {target_variable}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list for effective feature engineering.
            """,
            input_variables=[
                "user_instructions",
                "target_variable",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary_str = _summarize_df_for_prompt(df)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "target_variable": state.get("target_variable"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str,
        })

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Feature Engineering Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def create_feature_engineer_code(state: GraphState):
        print("    * CREATE FEATURE ENGINEER CODE")

        if bypass_recommended_steps:
            all_datasets_summary_str = _summarize_df_for_prompt(pd.DataFrame(state.get("data_raw")))
            steps_for_prompt = DEFAULT_FEATURE_STEPS
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            steps_for_prompt = state.get("recommended_steps") or DEFAULT_FEATURE_STEPS

        feature_engineering_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Agent. Your job is to create a {function_name}() function that generates meaningful features.

            Recommended Steps:
            {recommended_steps}

            User Instructions:
            {user_instructions}

            Target Variable:
            {target_variable}

            Data Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that:
            1. Includes all imports inside the function
            2. Creates new features based on existing columns
            3. Returns the enhanced data with new features
            4. Handles different data types appropriately

            Function signature:
            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import ...
                ...
                return data_featured

            Focus on creating features that improve model performance and interpretability.
            """,
            input_variables=[
                "recommended_steps",
                "user_instructions",
                "target_variable",
                "all_datasets_summary",
                "function_name",
            ],
        )

        feature_engineering_agent = feature_engineering_prompt | llm | PythonOutputParser()

        response = feature_engineering_agent.invoke({
            "recommended_steps": steps_for_prompt,
            "user_instructions": state.get("user_instructions"),
            "target_variable": state.get("target_variable"),
            "all_datasets_summary": all_datasets_summary_str,
            "function_name": function_name,
        })

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
            "feature_engineer_function": response,
            "feature_engineer_function_path": file_path,
            "feature_engineer_file_name": file_name_2,
            "feature_engineer_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
            "recommended_steps": steps_for_prompt,
        }

    def execute_feature_engineer_code(state: GraphState):
        print("    * EXECUTE FEATURE ENGINEER CODE (SANDBOXED)")

        result, error = run_code_sandboxed_subprocess(
            code_snippet=state.get("feature_engineer_function"),
            function_name=state.get("feature_engineer_function_name"),
            data=state.get("data_raw"),
            timeout=10,
            memory_limit_mb=512,
        )

        if error is None:
            try:
                data_featured = result if isinstance(result, dict) else {}
            except Exception as e:
                return {"data_featured": {}, "feature_engineer_error": f"Error processing featured data: {e!s}"}
            else:
                return {"data_featured": data_featured}
        else:
            return {"data_featured": {}, "feature_engineer_error": f"Feature engineering execution error: {error}"}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("recommend_feature_steps", recommend_feature_steps)
    workflow.add_node("create_feature_engineer_code", create_feature_engineer_code)
    workflow.add_node("execute_feature_engineer_code", execute_feature_engineer_code)
    workflow.add_node(
        "report_agent_outputs",
        lambda state: node_func_report_agent_outputs(
            state,
            [
                "recommended_steps",
                "feature_engineer_function",
                "feature_engineer_function_path",
                "feature_engineer_function_name",
                "feature_engineer_error",
                "data_featured",
            ],
        ),
    )

    workflow.add_edge(START, "recommend_feature_steps")
    workflow.add_edge("recommend_feature_steps", "create_feature_engineer_code")
    workflow.add_edge("create_feature_engineer_code", "execute_feature_engineer_code")
    workflow.add_edge("execute_feature_engineer_code", "report_agent_outputs")
    workflow.add_edge("report_agent_outputs", END)

    return workflow.compile()
