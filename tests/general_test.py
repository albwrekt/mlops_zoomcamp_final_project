"""
This file contains various tests for the project.
"""
# Data manipulation libraries
import pandas as pd
import numpy as np

# system path
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./ml_model_deployment")
# This brings in example data.
from utilities import *

# Bring in the model load and predict function
from deploy_model import predict

# random sampling for integer indices
from random import randint

# Requests for integration test
import requests


# Sample function
def add(a, b):
    return a + b


# Test the pre-commit framework to validate that unit testing works in the first place.
# This test is for set up only, and NOT MEANT TO CONTRIBUTE TO PROJECT SCORING. Thanks! -Eric
def test_add():
    assert add(2, 3) == 5
    assert add(3, 4) != 8


# Unit Test for small functionality
def test_specific_functionality():
    """
    Test the specific functionality of the
    """
    pass


# Unit Test of the model
def test_latest_model_prediction():
    """
    Grab the latest model, and verify the prediction is a valid numeric output.
    """
    # Generate an example feature
    example_df = generate_single_feature()
    # Now pass to the model prediction and return the output predictions.
    pred_df = predict(example_df)
    # print prediction dataframe
    print(pred_df)
    # Assert that pred_df exists
    assert pred_df is not None
    # Asssert that the pred_df is not empty
    assert pred_df.shape[0] > 0
    # Assert the predictions column is present and valid for all return entries
    assert all([0 < i <= 10000 for i in pred_df])  # unreasonably large number


# If deployed on local machine
deployed = False
if deployed:
    # Integration test of the deployed endpoint.
    def test_latest_deployed_model_prediction():
        """
        Test the api when deployed that serves the model prediction.
        """
        # Endpoint for test - default fast api test endpoint
        endpoint = "http://127.0.0.1:8000/"
        # Generate a single feature
        example_df = generate_single_feature()
        # Remove the label
        example_df = example_df[
            [col for col in example_df.columns if col != "ride_count"]
        ]
        # Convert boolean types to integers
        bool_cols = example_df.select_dtypes(bool).columns.tolist()
        # For every boolean column, convert it to string
        for bc in bool_cols:
            # Apply the transform to string
            example_df[bc] = example_df[bc].astype(np.float64)
        # Convert the pandas dataframe of one row to dictionary
        example_dict = {col: example_df[col][0] for col in example_df.columns}
        # Create post request to the API.
        result = requests.post(
            url=endpoint,  # Endpoint to contact.
            json=example_dict,  # Data passed through
        )
        # Wait for the return.
        print(result.json())


if __name__ == "__main__":
    test_latest_model_prediction()
