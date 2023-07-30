"""
This file contains the utilities for accessing and leveraging universal utilities in the directory.
"""
# Bring in the system path
import sys

# Libraries for enumeration
from enum import Enum, IntEnum, StrEnum

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
# model name
ride_count_non_holiday_model_name = "Ride Count Non-Holiday"


# Data Quality level enumeration for strings
class DataQualityEnum(StrEnum):
    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


# Experiment enumeration
class ExperimentEnum(StrEnum):
    RIDE_COUNT_NON_HOLIDAY = "ride_count_non_holiday"
