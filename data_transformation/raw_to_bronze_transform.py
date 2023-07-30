"""
File to enter the csv file data into the bronze format.
"""

# Parent path for utilities
import sys

# File utilities
from glob import glob

import numpy as np

# Import libraries for transformation
import pandas as pd

# Prefect utilities
from prefect import flow, task

sys.path.insert(0, "./")
# Utility libraries - this is needed for the system path integration
from utilities import *


# Method for transformation
@task(name="Raw to Bronze Transform")
def raw_to_bronze_transform():
    """
    This method transforms raw data into the bronze format for all tables in data directory.
    This assumes CSV input and converted to parquet.
    """
    # Bring in the data directory for raw data input.
    trips_df = pd.read_csv(
        data_directory_format.format(DataQualityEnum.RAW, "austin_bikeshare_trips.csv")
    )
    stations_df = pd.read_csv(
        data_directory_format.format(
            DataQualityEnum.RAW, "austin_bikeshare_stations.csv"
        )
    )
    # Apply the start nomenclature to id's
    stations_df.columns = ["start_" + col for col in stations_df.columns]
    # Merge the start station ID.
    merge_df = pd.merge(
        left=trips_df,
        right=stations_df,
        how="inner",
        left_on="start_station_id",
        right_on="start_station_id",
        suffixes=["_base", "_start"],
    )
    # Apply the start nomenclature to id's
    stations_df.columns = ["end" + col[col.index("_") :] for col in stations_df.columns]
    # Merge the end station ID.
    merge_df = pd.merge(
        left=merge_df,
        right=stations_df,
        how="inner",
        left_on="end_station_id",
        right_on="end_station_id",
        suffixes=["_base", "_end"],
    )
    # Save to parquet format.
    merge_df.to_parquet(
        data_directory_format.format(
            DataQualityEnum.BRONZE,
            data_pipeline_file_format.format(DataQualityEnum.BRONZE),
        )
    )
