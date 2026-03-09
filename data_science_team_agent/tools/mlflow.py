"""MLflow experiment tracking utilities."""

from typing import Any

from langchain.tools import tool  # type: ignore[import]

# MLflow is optional - handle import gracefully
try:
    import mlflow  # type: ignore[import]
    import mlflow.pytorch  # type: ignore[import]
    import mlflow.sklearn  # type: ignore[import]
    import mlflow.tensorflow  # type: ignore[import]
    from mlflow.tracking import MlflowClient  # type: ignore[import]

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. MLflow tools will not be available.")


@tool
def create_mlflow_experiment(
    experiment_name: str,
    artifact_location: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a new MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    artifact_location : str, optional
        Location to store artifacts. Defaults to None.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and experiment info

    """
    if not MLFLOW_AVAILABLE:
        return "MLflow is not available. Please install mlflow package.", {}

    try:
        experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)

        message = f"MLflow experiment '{experiment_name}' created successfully"

    except Exception as e:
        error_message = f"Error creating MLflow experiment: {e!s}"
        return error_message, {"error": error_message}
    else:
        return message, {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "artifact_location": artifact_location,
        }


@tool
def log_experiment_to_mlflow(  # noqa: C901 - complex logging is intentional
    model_data: dict[str, Any],
    metrics: dict[str, float],
    parameters: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    experiment_name: str = "Default",
    run_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Log an experiment to MLflow.

    Parameters
    ----------
    model_data : Dict[str, Any]
        Model or model-related data
    metrics : Dict[str, float]
        Dictionary of metrics to log
    parameters : Dict[str, Any], optional
        Dictionary of parameters to log
    artifacts : Dict[str, str], optional
        Dictionary of artifact file paths to log
    experiment_name : str, optional
        Name of the experiment. Defaults to "Default".
    run_name : str, optional
        Name of the run. Defaults to None.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and run info

    """
    if not MLFLOW_AVAILABLE:
        return "MLflow is not available. Please install mlflow package.", {}

    try:
        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Start run
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            if parameters:
                for param_name, param_value in parameters.items():
                    mlflow.log_param(param_name, param_value)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model if it's a scikit-learn model
            if "sklearn_model" in model_data:
                mlflow.sklearn.log_model(model_data["sklearn_model"], "model")
            elif "model_object" in model_data:
                # Try to log as generic model
                mlflow.log_artifact(model_data["model_object"], "model")

            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)

            run_id = run.info.run_id

        message = f"Experiment logged to MLflow successfully. Run ID: {run_id}"

    except Exception as e:
        error_message = f"Error logging to MLflow: {e!s}"
        return error_message, {"error": error_message}
    else:
        return message, {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "metrics": metrics,
            "parameters": parameters or {},
        }


@tool
def get_mlflow_run_info(
    run_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Get information about an MLflow run.

    Parameters
    ----------
    run_id : str, optional
        ID of the run. If None, gets the latest run.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and run information

    """
    if not MLFLOW_AVAILABLE:
        return "MLflow is not available. Please install mlflow package.", {}

    try:
        client = MlflowClient()

        if run_id:
            run = client.get_run(run_id)
        else:
            # Get latest run
            runs = client.search_runs(order_by=["start_time DESC"], max_results=1)
            if runs:
                run = runs[0]
            else:
                return "No runs found in MLflow.", {}

        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }

        message = f"Run information retrieved for: {run.info.run_id}"

    except Exception as e:
        error_message = f"Error getting MLflow run info: {e!s}"
        return error_message, {"error": error_message}
    else:
        return message, run_info


@tool
def list_mlflow_experiments() -> tuple[str, dict[str, Any]]:
    """List all MLflow experiments.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and list of experiments

    """
    if not MLFLOW_AVAILABLE:
        return "MLflow is not available. Please install mlflow package.", {}

    try:
        experiments = mlflow.search_experiments()

        experiment_list = []
        for exp in experiments:
            experiment_list.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": exp.creation_time,
            })

        message = f"Found {len(experiment_list)} experiments"

        return message, {"experiments": experiment_list, "count": len(experiment_list)}

    except Exception as e:
        error_message = f"Error listing MLflow experiments: {e!s}"
        return error_message, {"error": error_message}


@tool
def log_model_to_mlflow(
    model_object: Any,
    model_name: str,
    model_type: str = "sklearn",
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Log a model to MLflow model registry.

    Parameters
    ----------
    model_object : Any
        The trained model object
    model_name : str
        Name for the model
    model_type : str, optional
        Type of model ('sklearn', 'pytorch', 'tensorflow'). Defaults to 'sklearn'.
    artifact_path : str, optional
        Path to save model artifacts. Defaults to 'model'.
    registered_model_name : str, optional
        Name to register the model in MLflow registry. Defaults to None.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and model info

    """
    if not MLFLOW_AVAILABLE:
        return "MLflow is not available. Please install mlflow package.", {}

    try:
        with mlflow.start_run() as run:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model_object, artifact_path=artifact_path, registered_model_name=registered_model_name
                )
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(
                    model_object, artifact_path=artifact_path, registered_model_name=registered_model_name
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model_object, artifact_path=artifact_path, registered_model_name=registered_model_name
                )
            else:
                # Generic logging
                mlflow.log_artifact(model_object, artifact_path)

            run_id = run.info.run_id

        message = f"Model '{model_name}' logged to MLflow successfully"

    except Exception as e:
        error_message = f"Error logging model to MLflow: {e!s}"
        return error_message, {"error": error_message}
    else:
        return message, {
            "run_id": run_id,
            "model_name": model_name,
            "model_type": model_type,
            "registered_model_name": registered_model_name,
        }
