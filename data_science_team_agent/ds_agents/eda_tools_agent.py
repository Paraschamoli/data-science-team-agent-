"""EDA tools agent for exploratory data analysis operations.

Provides specialized agent capabilities for performing comprehensive
exploratory data analysis with automated statistical analysis
and visualization recommendations.
"""

import operator
from collections.abc import Sequence
from typing import Annotated

import pandas as pd
from langchain_core.messages import BaseMessage  # type: ignore[import]
from langgraph.graph import END, START  # type: ignore[import]
from typing_extensions import TypedDict

from data_science_team_agent.templates import (
    BaseAgent,
    create_coding_agent_graph,
)
from data_science_team_agent.tools.eda import (
    analyze_missing_values,
    correlation_analysis,
    detect_outliers,
)
from data_science_team_agent.utils.regex import (
    format_agent_name,
)

AGENT_NAME = "eda_tools_agent"


class EDAToolsAgent(BaseAgent):
    """Agent for exploratory data analysis and statistical operations."""

    def __init__(self, model, checkpointer=None):
        """Initialize the EDA tools agent.

        Args:
            model: The language model to use.
            checkpointer: Checkpointer for state management. Defaults to None.
        """
        self._params = {
            "model": model,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        self.response = None
        return make_eda_tools_agent(**self._params)

    def invoke_agent(self, user_instructions=None, data_raw=None, max_retries=3, retry_count=0, **kwargs):
        """Execute the agent workflow.

        Args:
            user_instructions: Optional user instructions. Defaults to None.
            data_raw: Optional raw pandas DataFrame. Defaults to None.
            max_retries: Maximum number of retries. Defaults to 3.
            retry_count: Current retry count. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated workflow state.
        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None


def make_eda_tools_agent(model, checkpointer=None):
    """Create an EDA tools agent.

    Args:
        model: The language model to use.
        checkpointer: Checkpointer for state management. Defaults to None.

    Returns:
        Compiled EDA tools agent graph.
    """

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        eda_report: str
        missing_analysis: str
        correlation_analysis: str
        outlier_analysis: str
        max_retries: int
        retry_count: int

    def generate_eda_report(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * GENERATE EDA REPORT")

        data_raw = state.get("data_raw")
        if not data_raw:
            return {"eda_report": "No data provided for EDA analysis"}

        df = pd.DataFrame(data_raw)

        # Generate comprehensive EDA report
        report_parts = []
        report_parts.append("# EXPLORATORY DATA ANALYSIS REPORT")
        report_parts.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        report_parts.append("")

        # Column information
        report_parts.append("## Column Information")
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            unique = df[col].nunique()

            report_parts.append(f"**{col}**")
            report_parts.append(f"- Type: {dtype}")
            report_parts.append(f"- Missing: {missing} ({missing_pct:.1f}%)")
            report_parts.append(f"- Unique values: {unique}")

            if df[col].dtype in ["int64", "float64"]:
                report_parts.append(f"- Range: {df[col].min():.2f} to {df[col].max():.2f}")
                report_parts.append(f"- Mean: {df[col].mean():.2f}")

            report_parts.append("")

        # Missing values analysis
        missing_analysis = analyze_missing_values(data_raw)
        report_parts.append("## Missing Values Analysis")
        report_parts.append(missing_analysis)

        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            corr_analysis = correlation_analysis(data_raw)
            report_parts.append("## Correlation Analysis")
            report_parts.append(corr_analysis)

        # Outlier detection
        outlier_analysis = detect_outliers(data_raw)
        report_parts.append("## Outlier Analysis")
        report_parts.append(outlier_analysis)

        return {"eda_report": "\n".join(report_parts)}

    def report_outputs(state: GraphState):
        print("    * REPORT EDA OUTPUTS")

        report = {
            "report_title": "EDA Analysis Report",
            "eda_report": state.get("eda_report", ""),
            "missing_analysis": state.get("missing_analysis", ""),
            "correlation_analysis": state.get("correlation_analysis", ""),
            "outlier_analysis": state.get("outlier_analysis", ""),
        }

        return {"agent_outputs": report}

    # Create workflow
    nodes = {
        "generate_eda_report": generate_eda_report,
        "report_outputs": report_outputs,
    }

    edges = [
        (START, "generate_eda_report"),
        ("generate_eda_report", "report_outputs"),
        ("report_outputs", END),
    ]

    return create_coding_agent_graph(
        nodes=nodes, edges=edges, state_class=GraphState, entry_point="generate_eda_report", checkpointer=checkpointer
    )
