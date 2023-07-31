"""
This file registers the desired model as the current model for the Registered Model in registry.
"""
# MLOps utilities
from mlflow import MlflowClient
import mlflow

# Prefect server utilities
from prefect import flow, task

# Project uilities
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./ml_experiments/ride_count_non_holiday")
from utilities import *

# Set tracking uri
mlflow.set_tracking_uri(mlflow_tracking_uri)


# Query the current models and register the current best model.
@flow(name="Register Best Model")
def register_best_model(experiment_id: int = 1):
    """
    Register the best performing model that is not overfitting from the mlflow registry.
    """
    # Start by querying all runs from the experiment_id
    client = MlflowClient(mlflow_tracking_uri)
    # Query the runs for the experiment.
    ml_runs = client.search_runs(experiment_ids=[experiment_id])
    # Parse the runs into viable usage.
    useful_runs = dict()
    # for each run
    for run in ml_runs:
        # Calculate difference in MAE as trigger here.
        # diff = (
        #     run.data.metrics["Testing Mean Absolute Error"]
        #     - run.data.metrics["Training Mean Absolute Error"]
        # )
        # If difference is less than threshold, most likely not overfit.
        useful_runs[run.data.metrics["Testing Mean Absolute Error"]] = run
    # Sort the output by the keys
    best_run = sorted(list(useful_runs.keys()))[0]
    best_run = useful_runs[best_run]
    # Now grab the Run ID for this model.
    best_run_id = best_run.data.tags["mlflow.log-model.history"]
    best_run_id = best_run_id[: best_run_id.index(",") - 1]
    best_run_id = best_run_id[best_run_id.rindex('"') + 1 :]
    # Get the latest version number of the model
    latest_version = client.get_latest_versions(
        name=ride_count_non_holiday_model_name, stages=["Production"]
    )
    # Test the rmse of the current best production.
    prod_run = client.get_run(run_id=latest_version[0].run_id)
    # If the prod run has a worse MAE than the new run
    if (
        prod_run.data.metrics["Testing Mean Absolute Error"]
        > best_run.data.metrics["Testing Mean Absolute Error"]
    ):
        # Grab the artifact path
        artifact_path = best_run.data.tags["mlflow.log-model.history"]
        artifact_path = artifact_path[
            artifact_path.index('"artifact_path": "') + len('"artifact_path": "') :
        ]
        artifact_path = artifact_path[: artifact_path.index('"')]
        # Register the new model.
        run_string = f"runs:/{best_run_id}/artifacts/{artifact_path}"
        registered_version = mlflow.register_model(
            run_string,
            ride_count_non_holiday_model_name,
        )
        # Transition the current model out to archive.
        client.transition_model_version_stage(
            name=ride_count_non_holiday_model_name,
            version=int(latest_version[0].version),
            stage="Archived",
        )
        # Promote the version to staging.
        client.transition_model_version_stage(
            name=ride_count_non_holiday_model_name,
            version=int(registered_version.version),
            stage="Staging",
        )
        # Run a test here eventually.

        # Promote to Production if test passes.
        client.transition_model_version_stage(
            name=ride_count_non_holiday_model_name,
            version=int(registered_version.version),
            stage="Production",
        )
        # If it doesn't pass, re-promote the previous to staging and send the other to None


# This function is dedicated to reverting to the previous model version if errors arise
@task(name="Reverting Model to Previous Version")
def revert_previous_model_version():
    """
    This function focuses on reverting to the previous model in the registry
    """
    # mlflow client
    client = MlflowClient(mlflow_tracking_uri)
    # Search for the latest version
    latest_version = int(
        client.get_latest_versions(
            name=ride_count_non_holiday_model_name, stages=["Production"]
        )[0].version
    )
    # Subtract to get next lowest version
    next_version = latest_version - 1
    # Now archive the current version
    client.transition_model_version_stage(
        name=ride_count_non_holiday_model_name, version=latest_version, stage="Archived"
    )
    # Now promote the old version back to Production.
    client.transition_model_version_stage(
        name=ride_count_non_holiday_model_name, version=next_version, stage="Production"
    )


# main
if __name__ == "__main__":
    revert_previous_model_version()
