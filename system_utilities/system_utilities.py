"""
This file contains the Threading needed to start all involved servers
"""
# operating system utilities
import os
import subprocess
from multiprocessing import process

# Threading utilities
from threading import Thread


# Thread for prefect server start
def start_prefect_server():
    """
    Start the local prefect server
    """
    # Start server with os command
    subprocess.run(["prefect", "server", "start"])
    # Sit until process is killed.
    while True:
        pass
