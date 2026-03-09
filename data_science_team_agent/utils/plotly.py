import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any


def plotly_from_dict(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a Plotly figure from a dictionary specification."""
    chart_type = plot_dict.get('chart_type', 'scatter')
    
    if chart_type == 'scatter':
        return create_scatter_plot(plot_dict)
    elif chart_type == 'bar':
        return create_bar_plot(plot_dict)
    elif chart_type == 'line':
        return create_line_plot(plot_dict)
    elif chart_type == 'histogram':
        return create_histogram(plot_dict)
    elif chart_type == 'box':
        return create_box_plot(plot_dict)
    elif chart_type == 'heatmap':
        return create_heatmap(plot_dict)
    else:
        # Default to scatter plot
        return create_scatter_plot(plot_dict)


def create_scatter_plot(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a scatter plot."""
    data = plot_dict.get('data', {})
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    x_col = plot_dict.get('x_column')
    y_col = plot_dict.get('y_column')
    color_col = plot_dict.get('color_column')
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=plot_dict.get('title', 'Scatter Plot'),
        labels=plot_dict.get('labels', {})
    )
    
    return fig


def create_bar_plot(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a bar plot."""
    data = plot_dict.get('data', {})
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    x_col = plot_dict.get('x_column')
    y_col = plot_dict.get('y_column')
    color_col = plot_dict.get('color_column')
    
    fig = px.bar(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=plot_dict.get('title', 'Bar Plot'),
        labels=plot_dict.get('labels', {})
    )
    
    return fig


def create_line_plot(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a line plot."""
    data = plot_dict.get('data', {})
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    x_col = plot_dict.get('x_column')
    y_col = plot_dict.get('y_column')
    color_col = plot_dict.get('color_column')
    
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=plot_dict.get('title', 'Line Plot'),
        labels=plot_dict.get('labels', {})
    )
    
    return fig


def create_histogram(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a histogram."""
    data = plot_dict.get('data', {})
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    x_col = plot_dict.get('x_column')
    color_col = plot_dict.get('color_column')
    
    fig = px.histogram(
        df, 
        x=x_col, 
        color=color_col,
        title=plot_dict.get('title', 'Histogram'),
        labels=plot_dict.get('labels', {})
    )
    
    return fig


def create_box_plot(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a box plot."""
    data = plot_dict.get('data', {})
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    x_col = plot_dict.get('x_column')
    y_col = plot_dict.get('y_column')
    color_col = plot_dict.get('color_column')
    
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=plot_dict.get('title', 'Box Plot'),
        labels=plot_dict.get('labels', {})
    )
    
    return fig


def create_heatmap(plot_dict: Dict[str, Any]) -> go.Figure:
    """Create a heatmap."""
    data = plot_dict.get('data', {})
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # For heatmap, we typically want correlation matrix
    if plot_dict.get('correlation', False):
        df = df.select_dtypes(include=['number']).corr()
    
    fig = px.imshow(
        df, 
        title=plot_dict.get('title', 'Heatmap'),
        labels=plot_dict.get('labels', {})
    )
    
    return fig
