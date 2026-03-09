import pandas as pd
import numpy as np
from langchain.tools import tool
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px


@tool
def generate_eda_report(
    data: Dict[str, Any],
    target_column: str = None,
) -> str:
    """
    Generate a comprehensive exploratory data analysis report.
    
    Parameters:
    ----------
    data : Dict[str, Any]
        Data to analyze (should be convertible to DataFrame)
    target_column : str, optional
        Target column for supervised analysis
    
    Returns:
    -------
    str
        EDA report as formatted text
    """
    print("    * Tool: generate_eda_report")
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            return "Cannot generate EDA report for empty DataFrame"
        
        report = []
        report.append("# EXPLORATORY DATA ANALYSIS REPORT")
        report.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        report.append("")
        
        # Column information
        report.append("## Column Information")
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            unique = df[col].nunique()
            
            report.append(f"**{col}**")
            report.append(f"- Type: {dtype}")
            report.append(f"- Missing: {missing} ({missing_pct:.1f}%)")
            report.append(f"- Unique values: {unique}")
            
            if df[col].dtype in ['int64', 'float64']:
                report.append(f"- Range: {df[col].min():.2f} to {df[col].max():.2f}")
                report.append(f"- Mean: {df[col].mean():.2f}")
            
            report.append("")
        
        # Missing values analysis
        missing_analysis = analyze_missing_values(data)
        report.append("## Missing Values Analysis")
        report.append(missing_analysis)
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_analysis = correlation_analysis(data)
            report.append("## Correlation Analysis")
            report.append(corr_analysis)
        
        # Target variable analysis (if provided)
        if target_column and target_column in df.columns:
            report.append("## Target Variable Analysis")
            target_series = df[target_column]
            
            if target_series.dtype in ['int64', 'float64']:
                report.append(f"- Distribution: Mean={target_series.mean():.2f}, Std={target_series.std():.2f}")
                report.append(f"- Range: {target_series.min():.2f} to {target_series.max():.2f}")
            else:
                value_counts = target_series.value_counts()
                report.append("- Value Distribution:")
                for val, count in value_counts.head(10).items():
                    pct = (count / len(target_series)) * 100
                    report.append(f"  {val}: {count} ({pct:.1f}%)")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating EDA report: {str(e)}"


@tool
def analyze_missing_values(
    data: Dict[str, Any],
) -> str:
    """
    Analyze missing values in the dataset.
    
    Parameters:
    ----------
    data : Dict[str, Any]
        Data to analyze
    
    Returns:
    -------
    str
        Missing values analysis report
    """
    print("    * Tool: analyze_missing_values")
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            return "No data to analyze for missing values"
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Create missing values report
        report = []
        report.append("### Missing Values Summary")
        report.append(f"Total rows: {len(df)}")
        report.append("")
        
        # Columns with missing values
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) == 0:
            report.append("✅ No missing values found in the dataset")
        else:
            report.append("Columns with missing values:")
            for col, count in missing_cols.sort_values(ascending=False).items():
                pct = missing_percentages[col]
                report.append(f"- **{col}**: {count} missing ({pct:.1f}%)")
            
            # Missing values patterns
            total_missing = missing_counts.sum()
            total_cells = len(df) * len(df.columns)
            overall_missing_pct = (total_missing / total_cells) * 100
            
            report.append("")
            report.append(f"Overall missing data: {overall_missing_pct:.1f}%")
            
            if overall_missing_pct > 50:
                report.append("⚠️  High missing data percentage detected")
            elif overall_missing_pct > 20:
                report.append("⚠️  Moderate missing data percentage detected")
            else:
                report.append("✅ Low missing data percentage")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error analyzing missing values: {str(e)}"


@tool
def correlation_analysis(
    data: Dict[str, Any],
    method: str = "pearson",
) -> str:
    """
    Perform correlation analysis on numeric columns.
    
    Parameters:
    ----------
    data : Dict[str, Any]
        Data to analyze
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall'). Defaults to 'pearson'.
    
    Returns:
    -------
    str
        Correlation analysis report
    """
    print(f"    * Tool: correlation_analysis | method: {method}")
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            return "No data to analyze for correlations"
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return "Need at least 2 numeric columns for correlation analysis"
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        report = []
        report.append(f"### Correlation Matrix ({method.title()} Method)")
        report.append("")
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    strong_correlations.append((col1, col2, corr_val))
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if strong_correlations:
            report.append("#### Strong Correlations (|r| > 0.7):")
            for col1, col2, corr_val in strong_correlations:
                direction = "positive" if corr_val > 0 else "negative"
                report.append(f"- **{col1}** ↔ **{col2}**: {corr_val:.3f} ({direction})")
        else:
            report.append("No strong correlations found (|r| > 0.7)")
        
        report.append("")
        report.append("#### Correlation Matrix:")
        
        # Format correlation matrix
        corr_str = corr_matrix.round(3).to_string()
        report.append("```\n" + corr_str + "\n```")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error performing correlation analysis: {str(e)}"


@tool
def detect_outliers(
    data: Dict[str, Any],
    method: str = "iqr",
    threshold: float = 1.5,
) -> str:
    """
    Detect outliers in numeric columns.
    
    Parameters:
    ----------
    data : Dict[str, Any]
        Data to analyze
    method : str, optional
        Outlier detection method ('iqr', 'zscore'). Defaults to 'iqr'.
    threshold : float, optional
        Threshold for outlier detection. Defaults to 1.5 for IQR method.
    
    Returns:
    -------
    str
        Outlier detection report
    """
    print(f"    * Tool: detect_outliers | method: {method}")
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            return "No data to analyze for outliers"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return "No numeric columns found for outlier detection"
        
        report = []
        report.append("### Outlier Detection Report")
        report.append(f"Method: {method.title()}")
        report.append("")
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = series[z_scores > threshold]
            else:
                continue
            
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(series)) * 100
            
            report.append(f"**{col}**:")
            report.append(f"- Outliers detected: {outlier_count} ({outlier_percentage:.1f}%)")
            
            if outlier_count > 0:
                report.append(f"- Outlier range: {outliers.min():.2f} to {outliers.max():.2f}")
                if method == "iqr":
                    report.append(f"- IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            report.append("")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error detecting outliers: {str(e)}"
