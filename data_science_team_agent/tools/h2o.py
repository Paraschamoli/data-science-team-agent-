import pandas as pd
import numpy as np
from langchain.tools import tool
from typing import Dict, Any, List, Optional, Tuple

# H2O-3 is optional - handle import gracefully
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("Warning: H2O-3 not installed. H2O tools will not be available.")


@tool
def initialize_h2o(
    max_mem_size: str = "2G",
    nthreads: int = -1,
) -> str:
    """
    Initialize H2O cluster.
    
    Parameters:
    ----------
    max_mem_size : str, optional
        Maximum memory size for H2O cluster. Defaults to "2G".
    nthreads : int, optional
        Number of threads. -1 means use all available. Defaults to -1.
    
    Returns:
    -------
    str
        Status message
    """
    if not H2O_AVAILABLE:
        return "H2O is not available. Please install h2o package."
    
    try:
        h2o.init(max_mem_size=max_mem_size, nthreads=nthreads)
        return f"H2O cluster initialized successfully with max memory: {max_mem_size}"
    except Exception as e:
        return f"Error initializing H2O: {str(e)}"


@tool
def train_h2o_model(
    data: Dict[str, Any],
    target_column: str,
    model_type: str = "auto",
    max_runtime_secs: int = 60,
    max_models: int = 20,
    sort_metric: str = "AUTO",
) -> Tuple[str, Dict[str, Any]]:
    """
    Train an H2O AutoML model.
    
    Parameters:
    ----------
    data : Dict[str, Any]
        Training data
    target_column : str
        Target variable column name
    model_type : str, optional
        Model type ('auto', 'regression', 'classification'). Defaults to 'auto'.
    max_runtime_secs : int, optional
        Maximum runtime in seconds. Defaults to 60.
    max_models : int, optional
        Maximum number of models to train. Defaults to 20.
    sort_metric : str, optional
        Sorting metric for model selection. Defaults to 'AUTO'.
    
    Returns:
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and model information
    """
    if not H2O_AVAILABLE:
        return "H2O is not available. Please install h2o package.", {}
    
    try:
        df = pd.DataFrame(data)
        
        if target_column not in df.columns:
            return f"Target column '{target_column}' not found in data.", {}
        
        # Convert to H2O frame
        h2o_df = h2o.H2OFrame(df)
        
        # Identify predictors and response
        predictors = [col for col in df.columns if col != target_column]
        
        # Train AutoML model
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            sort_metric=sort_metric,
            seed=42
        )
        
        aml.train(x=predictors, y=target_column, training_frame=h2o_df)
        
        # Get leader model
        leader = aml.leader
        
        message = f"H2O AutoML model trained successfully. Best model: {leader.model_id}"
        
        return message, {
            "model_id": leader.model_id,
            "model_type": model_type,
            "performance": leader.model_performance().as_dict(),
            "leaderboard": aml.leaderboard.as_dict(),
            "training_time": aml.training_info()['duration']['msecs'] / 1000
        }
        
    except Exception as e:
        error_message = f"Error training H2O model: {str(e)}"
        return error_message, {"error": error_message}


@tool
def predict_with_h2o_model(
    data: Dict[str, Any],
    model_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Make predictions using a trained H2O model.
    
    Parameters:
    ----------
    data : Dict[str, Any]
        Data to make predictions on
    model_id : str
        ID of the trained H2O model
    
    Returns:
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and predictions
    """
    if not H2O_AVAILABLE:
        return "H2O is not available. Please install h2o package.", {}
    
    try:
        df = pd.DataFrame(data)
        h2o_df = h2o.H2OFrame(df)
        
        # Get the model
        model = h2o.get_model(model_id)
        
        # Make predictions
        predictions = model.predict(h2o_df)
        
        # Convert to pandas DataFrame
        pred_df = predictions.as_data_frame()
        
        message = f"Predictions generated successfully for {len(pred_df)} rows"
        
        return message, {
            "predictions": pred_df.to_dict(),
            "shape": pred_df.shape,
            "columns": list(pred_df.columns)
        }
        
    except Exception as e:
        error_message = f"Error making predictions: {str(e)}"
        return error_message, {"error": error_message}


@tool
def get_h2o_model_summary(
    model_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Get summary information about a trained H2O model.
    
    Parameters:
    ----------
    model_id : str
        ID of the trained H2O model
    
    Returns:
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and model summary
    """
    if not H2O_AVAILABLE:
        return "H2O is not available. Please install h2o package.", {}
    
    try:
        model = h2o.get_model(model_id)
        
        # Get model summary
        summary = {
            "model_id": model_id,
            "model_type": model.algo,
            "parameters": model.params.as_dict(),
            "performance": model.model_performance().as_dict(),
        }
        
        # Add variable importance if available
        if hasattr(model, 'varimp'):
            try:
                varimp = model.varimp()
                if varimp is not None:
                    summary["variable_importance"] = varimp.as_dict()
            except:
                pass
        
        message = f"Model summary retrieved for: {model_id}"
        
        return message, summary
        
    except Exception as e:
        error_message = f"Error getting model summary: {str(e)}"
        return error_message, {"error": error_message}


@tool
def shutdown_h2o() -> str:
    """
    Shutdown H2O cluster.
    
    Returns:
    -------
    str
        Status message
    """
    if not H2O_AVAILABLE:
        return "H2O is not available."
    
    try:
        h2o.cluster().shutdown()
        return "H2O cluster shutdown successfully"
    except Exception as e:
        return f"Error shutting down H2O: {str(e)}"
