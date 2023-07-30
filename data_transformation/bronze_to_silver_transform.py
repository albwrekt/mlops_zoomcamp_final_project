"""
This file contains functionality to bring data from bronze to silver functionality.
"""

# datetime for duration
from datetime import datetime, timedelta

# File utilities
from glob import glob

# Iter tools for permutation
from itertools import product

import numpy as np

# Import data manipulation libraries
import pandas as pd

# Prefect utilities
from prefect import flow, task

# Utilities for project
from utilities import *


# Bronze to silver workflow
@task(name="Bronze to Silver Transform")
def bronze_to_silver_transform():
    """
    Bronze to silver transformation for incoming dataset.
    This should be custom to the dataset for merges!
    Run the bronze level rule checks: no duplication, remove nulls, then apply bounds checking to the columns.
    """
    # Read in the bronze data directory file parquet
    bronze_df = pd.read_parquet(
        data_directory_format.format(
            DataQualityEnum.BRONZE,
            data_pipeline_file_format.format(DataQualityEnum.BRONZE),
        )
    )
    # Fill empty columns as needed
    bronze_df["subscriber_type"] = bronze_df["subscriber_type"].apply(
        lambda x: x if x != "" and x != np.nan else "Unknown Type"
    )
    # Convert the datetimes to individual components
    bronze_df["start_time"] = pd.to_datetime(bronze_df["start_time"])
    bronze_df["timedelta_duration"] = bronze_df["duration_minutes"].apply(
        lambda x: timedelta(minutes=x)
    )
    bronze_df["end_time"] = bronze_df["start_time"] + bronze_df["timedelta_duration"]
    # Drop the duration column
    bronze_df.drop("timedelta_duration", axis=1, inplace=True)
    # Columnarize them.
    for prefix in ["start", "end"]:
        # Create the columns
        bronze_df[f"{prefix}_day"] = bronze_df[f"{prefix}_time"].apply(lambda x: x.day)
        bronze_df[f"{prefix}_hour"] = bronze_df[f"{prefix}_time"].apply(
            lambda x: x.hour
        )
        bronze_df[f"{prefix}_minutes"] = bronze_df[f"{prefix}_time"].apply(
            lambda x: x.minute
        )
        bronze_df[f"{prefix}_seconds"] = bronze_df[f"{prefix}_time"].apply(
            lambda x: x.second
        )
    # Remove the nulls.
    bronze_df.dropna(inplace=True, axis=0)
    # Remove the duplicates of all rows
    bronze_df.drop_duplicates(inplace=True)
    # Drop extraneous columns that are not used for prediction.
    bronze_df.drop(
        [
            "checkout_time",
            "start_status",
            "end_status",
            "start_name",
            "end_name",
            "start_location",
            "end_location",
            "start_station_name",
            "end_station_name",
        ],
        axis=1,
        inplace=True,
    )
    # Greater than or equal to zero numeric bounds checking.
    for col in [
        "duration_minutes",
        "month",
        "end_station_id",
        "start_station_id",
        "bikeid",
        "trip_id",
        "year",
        "start_day",
        "start_hour",
        "start_minutes",
        "start_seconds",
    ]:
        # Process the function into the dataframe column.
        bronze_df = bronze_df[bronze_df[col] >= 0]
    # Write the data to the silver parquet folder
    bronze_df.to_parquet(
        data_directory_format.format(
            DataQualityEnum.SILVER,
            data_pipeline_file_format.format(DataQualityEnum.SILVER),
        )
    )
