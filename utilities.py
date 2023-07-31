"""
This file contains the utilities for accessing and leveraging universal utilities in the directory.
"""
# Bring in the system path
import sys

# Bring in random integer generation
from random import randint

# Data manipulation
import pandas as pd

# Libraries for enumeration
from enum import StrEnum

# Place the current directory into the system path for utilities and other imports
sys.path.insert(0, "./")

# Data directory formats
data_directory_format = "data/{0}/{1}"
# Data file format
data_pipeline_file_format = "{0}_data.parquet"
# Experiment file format
experiment_data_format = "data/experiments/{0}/{1}.parquet"
# tracking uri for mlflow
mlflow_tracking_uri = "http://127.0.0.1:5000"
# experiment name
experiment_name = "ride_count_non_holiday"
# model name
ride_count_non_holiday_model_name = "Ride Count Non-Holiday"


# Generate single point of data from the stored featurized data.
def generate_single_feature():
    """
    Grab a single feature that can be used to test out the model prediction in either case.
    """
    # Load in the featurized data for the test.
    location = experiment_data_format.format(experiment_name, "ml_input")
    # Bring data to dataframe
    feature_df = pd.read_parquet(location)
    # Parse out the desired label
    feature_df = feature_df[[col for col in feature_df.columns if col != "ride_count"]]
    # Randomly sample the data within
    random_sample = randint(0, feature_df.shape[0] - 1)
    # Grab this line only
    return feature_df.iloc[random_sample : random_sample + 1]


# Data Quality level enumeration for strings
class DataQualityEnum(StrEnum):
    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


# Experiment enumeration
class ExperimentEnum(StrEnum):
    RIDE_COUNT_NON_HOLIDAY = "ride_count_non_holiday"
