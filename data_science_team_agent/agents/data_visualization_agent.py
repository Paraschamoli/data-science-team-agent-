"""Data visualization agent for creating charts and plots."""

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

AGENT_NAME = "data_visualization_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


class DataVisualizationAgent(BaseAgent):
    """Agent responsible for creating data visualizations and charts."""

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_visualization.py",
        function_name="data_visualization",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer: Checkpointer = None,
    ):
        """Initialize the data visualization agent.

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
        data_raw: pd.DataFrame,
        user_instructions: str | None = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Execute the data visualization agent workflow.

        Args:
            data_raw: The raw data to visualize.
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
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        return None

    def _make_compiled_graph(self):
        self.response = None
        return make_data_visualization_agent(**self._params)


def make_data_visualization_agent(  # noqa: C901 - complex agent setup is intentional
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_visualization.py",
    function_name="data_visualization",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    """Create a data visualization agent for generating charts and plots.

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
        Compiled data visualization agent.

    """
    llm = model
    MAX_SUMMARY_COLUMNS = 30

    DEFAULT_VISUALIZATION_STEPS = format_recommended_steps(
        """
1. Analyze the data to determine appropriate chart types for different columns.
2. Create user-friendly titles and axis labels.
3. Apply consistent styling (template, font sizes, color themes).
4. Handle theme details (white background, base font size, line size, etc.).
        """,
        heading="# Recommended Visualization Steps:",
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
        recommended_steps: str
        data_raw: dict
        plot_data: dict
        all_datasets_summary: str
        data_visualization_function: str
        data_visualization_function_path: str
        data_visualization_file_name: str
        data_visualization_function_name: str
        data_visualization_error: str
        max_retries: int
        retry_count: int

    def recommend_visualization_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND VISUALIZATION STEPS")

        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Visualization Expert. Given the following information about data and user instructions,
            recommend a series of steps to create effective visualizations.

            General Visualization Guidelines:
            * Analyze data to determine appropriate chart types (bar, scatter, line, histogram, box, heatmap)
            * Create user-friendly titles and axis labels
            * Apply consistent styling and color themes
            * Handle theme details appropriately

            Custom Steps:
            * Analyze user instructions to understand specific visualization requirements
            * Recommend chart types that best represent the data and user needs
            * Consider data types (categorical vs numerical) for chart selection

            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list for creating effective visualizations.
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
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str,
        })

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Visualization Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def create_visualization_code(state: GraphState):
        print("    * CREATE VISUALIZATION CODE")

        if bypass_recommended_steps:
            all_datasets_summary_str = _summarize_df_for_prompt(pd.DataFrame(state.get("data_raw")))
            steps_for_prompt = DEFAULT_VISUALIZATION_STEPS
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            steps_for_prompt = state.get("recommended_steps") or DEFAULT_VISUALIZATION_STEPS

        data_visualization_prompt = PromptTemplate(
            template="""
            You are a Data Visualization Agent. Your job is to create a {function_name}() function that generates effective Plotly visualizations.

            Recommended Steps:
            {recommended_steps}

            User Instructions:
            {user_instructions}

            Data Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that:
            1. Includes all imports inside the function
            2. Creates appropriate Plotly charts based on data characteristics
            3. Returns a dictionary with chart specifications
            4. Uses descriptive titles and labels

            Function signature:
            def {function_name}(data_raw):
                import pandas as pd
                import plotly.graph_objects as go
                import plotly.express as px
                ...
                return plot_specifications

            The plot_specifications should be a dictionary with chart information that can be used with plotly_from_dict().
            """,
            input_variables=[
                "recommended_steps",
                "user_instructions",
                "all_datasets_summary",
                "function_name",
            ],
        )

        data_visualization_agent = data_visualization_prompt | llm | PythonOutputParser()

        response = data_visualization_agent.invoke({
            "recommended_steps": steps_for_prompt,
            "user_instructions": state.get("user_instructions"),
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
            "data_visualization_function": response,
            "data_visualization_function_path": file_path,
            "data_visualization_file_name": file_name_2,
            "data_visualization_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
            "recommended_steps": steps_for_prompt,
        }

    def execute_visualization_code(state: GraphState):
        print("    * EXECUTE VISUALIZATION CODE (SANDBOXED)")

        result, error = run_code_sandboxed_subprocess(
            code_snippet=state.get("data_visualization_function"),
            function_name=state.get("data_visualization_function_name"),
            data=state.get("data_raw"),
            timeout=10,
            memory_limit_mb=512,
        )

        if error is None:
            try:
                plot_data = result if isinstance(result, dict) else {}
            except Exception as e:
                return {"plot_data": {}, "data_visualization_error": f"Error processing plot data: {e!s}"}
            else:
                return {"plot_data": plot_data}
        else:
            return {"plot_data": {}, "data_visualization_error": f"Visualization execution error: {error}"}

    workflow = StateGraph(GraphState, checkpointer=checkpointer)
    workflow.add_node("recommend_visualization_steps", recommend_visualization_steps)
    workflow.add_node("create_visualization_code", create_visualization_code)
    workflow.add_node("execute_visualization_code", execute_visualization_code)
    workflow.add_node(
        "report_agent_outputs",
        lambda state: node_func_report_agent_outputs(
            state,
            [
                "recommended_steps",
                "data_visualization_function",
                "data_visualization_function_path",
                "data_visualization_function_name",
                "data_visualization_error",
                "plot_data",
            ],
        ),
    )

    workflow.add_edge(START, "recommend_visualization_steps")
    workflow.add_edge("recommend_visualization_steps", "create_visualization_code")
    workflow.add_edge("create_visualization_code", "execute_visualization_code")
    workflow.add_edge("execute_visualization_code", "report_agent_outputs")
    workflow.add_edge("report_agent_outputs", END)

    return workflow.compile()
