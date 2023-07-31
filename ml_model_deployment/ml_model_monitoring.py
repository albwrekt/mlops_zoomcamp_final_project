"""
This file is for monitoring the model performance and making a reversion if the performance isn't as expected.
"""
# Data manipulation
import pandas as pd
import numpy as np

# Add the system path
import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../tests/")
# Import utilities
from utilities import *

# MLOps utilities
from mlflow import MlflowClient
import mlflow.pyfunc

mlflow.set_tracking_uri(mlflow_tracking_uri)

# UUID random generation
from uuid import uuid4

# Import evidently and the data presets.
from evidently.test_suite import TestSuite
from evidently import ColumnMapping
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset


# Run the monitoring on the output dataset
def parse_input_dataset(input_df: pd.DataFrame):
    """
    Run the model predictions against the DataDriftTestPreset and log the results.
    """
    # Grab the reference dataset.
    ref_df = load_reference_dataset()
    # Drop ride_count
    ref_df = ref_df[[col for col in ref_df.columns if col != "ride_count"]]
    comp_df = input_df.copy()[[col for col in input_df.columns if col != "ride_count"]]
    # Now run the test suite and generate the report.
    ts = TestSuite(tests=[DataDriftTestPreset()])
    # Run the suite
    ts.run(reference_data=ref_df, current_data=comp_df)
    # Save the file to the folder.
    ts.save_html(
        filename=data_directory_format.format(
            "drift_reference", f"{uuid4()}_report.html"
        )
    )
