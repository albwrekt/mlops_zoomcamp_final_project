"""
This file contains the utilities for accessing and leveraging universal utilities in the directory.
"""
# Libraries for enumeration
from enum import Enum, IntEnum, StrEnum

# Data directory formats
data_directory_format = "./data/{0}/{1}"


# Data Quality level enumeration for strings
class DataQualityEnum(StrEnum):
    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
