"""
This file contains the experiment pipelines to be run.
"""
# Operating system utilities
import sys

# Prefect tasks andflows
from prefect import flow, task

sys.path.insert(0, "./")
sys.path.insert(0, "./ml_experiments/ride_count_non_holiday")
sys.path.insert(0, "ml_model_deployment/")
# Datetime
from datetime import datetime, timedelta

import numpy as np

# Data manipulation
import pandas as pd

# Featurization of the data.
from ride_count_non_holiday import *

# Utilities for the project
from utilities import *

# ML Model Registry
from register_desired_model import register_best_model


@flow(name="Experiment: Ride Count on Non-Holidays")
def experiment_ride_count_non_holidays():
    """
    Predict the ride count on non_holidays using this pipeline.
    """
    # Featurize the data per day.
    features_per_day_df = featurize_ride_count_non_holiday()
    # Featurize the data for the previous week.
    features_per_week_df = featurize_per_week_ride_count_non_holiday(
        features_per_day_df
    )
    # Featurize the data for the previos month
    features_per_month_df = featurize_per_month_ride_count_non_holiday(
        features_per_week_df
    )
    # Convert all numeric columns to floats
    num_cols = features_per_month_df.select_dtypes(include=np.number).columns.tolist()
    features_per_month_df[num_cols] = features_per_month_df[num_cols].astype(np.float64)
    # Run the experiment items on the featurized dataset
    ml_experiment_ride_count_non_holiday(features_per_month_df)
    # Register best model from the experiment
    experiment_id = 1
    register_best_model(experiment_id)


# Main
if __name__ == "__main__":
    experiment_ride_count_non_holidays()
