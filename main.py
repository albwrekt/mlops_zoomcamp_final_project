"""
This file will run the entire pipeline to develop a model.
"""

# Operating system utilities
import os
import sys

import numpy as np

# Import libraries
import pandas as pd

# Path insertion for the system utilities
sys.path.insert(0, "./system_utilities")
sys.path.insert(0, "./data_transformation")
sys.path.insert(0, "./ml_experiments")
# Threading and subprocess utilities
from multiprocessing import Process
from threading import Thread

# Data pipeline
from data_pipeline_flow import data_pipeline

# Featurization
from experiment_pipeline_flow import experiment_ride_count_non_holidays

# Prefect utilities
from prefect import flow, task

from system_utilities import start_prefect_server

# Utilities for the project
from utilities import *


# Main method for the pipeline
@flow(name="Full Pipeline")
def main_flow():
    """
    This runs the full code pipeline for generating the curated datasets, ML Model and results.
    """
    # Move onto the data pipeline transformations
    data_pipeline()
    # ML Experiment pipeline
    experiment_ride_count_non_holidays()


# main system method
def main():
    """
    This includes the system utilities wrapped around main utility
    """
    # Start the system utilities
    process_list = []
    # For each process function
    for func in [start_prefect_server]:
        # Make new process
        nproc = Process(target=func)
        # Start process
        nproc.start()
        # Add process to list
        process_list.append(nproc)
    # Main flow
    main_flow()
    # Terminate all processes when done
    for p in process_list:
        # Kill the process
        p.terminate()


# If main
if __name__ == "__main__":
    main()
