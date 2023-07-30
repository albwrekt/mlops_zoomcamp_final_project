"""
This file contains functionality to bring data from silver to gold functionality.
"""

# Datetime utilities
from datetime import datetime, timedelta

# File utilities
from glob import glob

import numpy as np

# Import data manipulation libraries
import pandas as pd

# Prefect orchestration utilities
from prefect import flow, task

# Utilities for project
from utilities import *


@task(name="Silver to Gold Transform")
def silver_to_gold_transform():
    """
    Silver to gold transformation for incoming dataset.
    This should be custom to the dataset for merges!
    """
    # Read in the data from the silver landing spot.
    silver_df = pd.read_parquet(
        data_directory_format.format(
            DataQualityEnum.SILVER,
            data_pipeline_file_format.format(DataQualityEnum.SILVER),
        )
    )

    # Featurize the day of the week
    silver_df["day_of_week"] = silver_df["start_time"].apply(lambda x: x.weekday())
    # Featurize week day vs. weekend
    silver_df["wday_vs_wend"] = silver_df["day_of_week"].apply(
        lambda x: "wday" if x < 5 else "wend"
    )
    # Featurize the season
    silver_df["season"] = silver_df["start_time"].apply(lambda x: x.month % 12 // 3 + 1)
    # Extended holiday list
    extended_holidays = []
    holiday_radius = 1
    # For the year range
    for y in range(int(silver_df["year"].min()), int(silver_df["year"].max() + 1)):
        # For the day month combo
        for m, d in list(zip([1, 2, 7, 10, 12, 12, 12], [1, 14, 4, 31, 24, 25, 31])):
            # make the datetime and radius events
            for d_i in range(-holiday_radius, holiday_radius + 1):
                # Add the event based on the radius
                extended_holidays.append(
                    datetime(year=y, month=m, day=d) + timedelta(d_i)
                )
    # Now filter the dataset against the extended holidays
    silver_df["holiday"] = silver_df["start_time"].apply(
        lambda x: 0
        if datetime(
            year=x.to_pydatetime().year,
            month=x.to_pydatetime().month,
            day=x.to_pydatetime().day,
        )
        not in extended_holidays
        else 1
    )
    # place the rides in order of start time for the final transformation
    silver_df = silver_df.sort_values(by=["start_time"], axis=0)
    # Create a date only column
    silver_df["start_date"] = silver_df["start_time"].apply(
        lambda x: datetime(
            year=x.to_pydatetime().year,
            month=x.to_pydatetime().month,
            day=x.to_pydatetime().day,
        )
    )
    # Write the data back out to gold landing spot.
    silver_df.to_parquet(
        data_directory_format.format(
            DataQualityEnum.GOLD, data_pipeline_file_format.format(DataQualityEnum.GOLD)
        )
    )
