"""
This file works to deploy the model as an API Endpoint for inference.
"""
# Library for API Endpoint
from fastapi import FastAPI, Request

# Data manipulation
import pandas as pd
import numpy as np

# Add the system path
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./tests")
# Import utilities
from utilities import *

# Import the testing utilities to validate the endpoint access and predictions
from general_test import (
    generate_single_feature,
)  # Single ML Ready feature saved during data pipelineing

# MLOps utilities for model registry and stuff.
from mlflow import MlflowClient
import mlflow.pyfunc

mlflow.set_tracking_uri(mlflow_tracking_uri)
# Requests to submit the data to the API
import requests


# Method for requesting the API Endpoint access to submit a prediction.
def api_submit_features_for_prediction(endpoint: str = "http://localhost:8000"):
    """
    Submit a feature against the running endpoint for validation of prediction.
    """
    # Generate a single feature
    example_df = generate_single_feature()
    # Remove the label
    example_df = example_df[[col for col in example_df.columns if col != "ride_count"]]
    # Convert the pandas dataframe of one row to dictionary
    example_dict = {col: example_df[col][0] for col in example_df.columns}
    # Create post request to the API.
    result = requests.post(
        url=endpoint,  # Endpoint to contact.
        data=example_dict,  # Data passed through
    )
    # Wait for the return.
    print(result.json())
