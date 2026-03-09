"""DataFrame analysis and summary utilities."""

from typing import Any

import numpy as np
import pandas as pd


def get_dataframe_summary(  # noqa: C901 - complex analysis is intentional
    df: pd.DataFrame | list[pd.DataFrame],
    max_rows: int = 1000,
    n_sample: int = 5,
    skip_stats: bool = False,
) -> list[str]:
    """Generate summary statistics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or List[pd.DataFrame]
        Single DataFrame or list of DataFrames to summarize
    n_sample : int, optional
        Number of sample rows to show. Defaults to 5.
    skip_stats : bool, optional
        Whether to skip statistical summary. Defaults to False.

    Returns
    -------
    List[str]
        List of summary strings for each DataFrame

    """
    summaries = []

    # Handle both single DataFrame and list of DataFrames
    dataframes = df if isinstance(df, list) else [df]

    for i, dataframe in enumerate(dataframes):
        if dataframe is None or dataframe.empty:
            summaries.append(f"DataFrame {i + 1}: Empty or None")
            continue

        summary_parts = [f"DataFrame {i + 1} Summary:"]

        # Basic info
        summary_parts.append(f"Shape: {dataframe.shape}")
        summary_parts.append(f"Columns: {list(dataframe.columns)}")

        # Data types
        dtype_counts = dataframe.dtypes.value_counts().to_dict()
        summary_parts.append("Data Types:")
        for dtype, count in dtype_counts.items():
            summary_parts.append(f"  {dtype}: {count}")

        # Sample data
        if n_sample > 0:
            summary_parts.append(f"\nSample Data (first {n_sample} rows):")
            sample_df = dataframe.head(n_sample)
            for _, row in sample_df.iterrows():
                summary_parts.append(f"  {dict(row)}")

        # Statistical summary
        if not skip_stats:
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_parts.append("\nNumeric Columns Summary:")
                for col in numeric_cols:
                    series = dataframe[col]
                    summary_parts.append(f"  {col}:")
                    summary_parts.append(f"    Mean: {series.mean():.2f}")
                    summary_parts.append(f"    Std: {series.std():.2f}")
                    summary_parts.append(f"    Min: {series.min()}")
                    summary_parts.append(f"    Max: {series.max()}")
                    summary_parts.append(f"    Missing: {series.isnull().sum()}")

            # Missing values summary
            missing_counts = dataframe.isnull().sum()
            if missing_counts.sum() > 0:
                summary_parts.append("\nMissing Values:")
                for col, count in missing_counts.items():
                    if count > 0:
                        pct = (count / len(dataframe)) * 100
                        summary_parts.append(f"  {col}: {count} ({pct:.1f}%)")

        summaries.append("\n".join(summary_parts))

    return summaries


def describe_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive description of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to describe

    Returns
    -------
    Dict[str, Any]
        Dictionary containing DataFrame description

    """
    if df is None or df.empty:
        return {"error": "DataFrame is empty or None"}

    description = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        description["numeric_summary"] = {}
        for col in numeric_cols:
            series = df[col]
            description["numeric_summary"][col] = {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "missing_count": series.isnull().sum(),
                "missing_percentage": (series.isnull().sum() / len(series)) * 100,
            }

    # Categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        description["categorical_summary"] = {}
        for col in categorical_cols:
            series = df[col]
            description["categorical_summary"][col] = {
                "unique_count": series.nunique(),
                "most_frequent": series.mode().iloc[0] if not series.mode().empty else None,
                "missing_count": series.isnull().sum(),
                "missing_percentage": (series.isnull().sum() / len(series)) * 100,
            }

    return description


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a DataFrame and return validation results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate

    Returns
    -------
    Dict[str, Any]
        Validation results

    """
    if df is None:
        return {"valid": False, "error": "DataFrame is None"}

    if df.empty:
        return {"valid": False, "error": "DataFrame is empty"}

    validation_results = {"valid": True, "warnings": [], "issues": []}

    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        validation_results["issues"].append(f"Duplicate columns: {duplicate_cols}")

    # Check for columns with all missing values
    all_missing_cols = df.columns[df.isnull().all()].tolist()
    if all_missing_cols:
        validation_results["warnings"].append(f"Columns with all missing values: {all_missing_cols}")

    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if memory_mb > 100:  # More than 100MB
        validation_results["warnings"].append(f"Large memory usage: {memory_mb:.1f}MB")

    return validation_results
