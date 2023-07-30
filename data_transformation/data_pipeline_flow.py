"""
This file contains the Prefect workflow for transforming data from raw to Gold ready data.
"""

# Parent path for utilities
import sys

import numpy as np

# Import libraries for data manipulation
import pandas as pd

sys.path.insert(0, "./")
from bronze_to_silver_transform import bronze_to_silver_transform

# Prefect orchestrator utilities
from prefect import flow, task

# Pipeline tasks
from raw_to_bronze_transform import raw_to_bronze_transform
from silver_to_gold_transform import silver_to_gold_transform

# Utilities for project
from utilities import *


# Main flow for Prefect
@flow(name="Data Transformation Pipeline")
def data_pipeline():
    """
    Raw to Gold transformation of the incoming dataset.
    """
    # Run the raw to bronze transform
    raw_to_bronze_transform()
    # Run the bronze to silver transform
    bronze_to_silver_transform()
    # Run the silver to gold transform
    silver_to_gold_transform()


# Main testing method
if __name__ == "__main__":
    data_pipeline()
