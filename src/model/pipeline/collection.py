"""
This module provides functionalities to load data from a database.

It includes a function to extract data from the Freq table
in the database and load it into a pandas DataFrame. This module is useful
for scenarios where data needs to be retrieved from a database for further
analysis or processing. It uses SQLAlchemy for executing database queries
and pandas for handling the data in a DataFrame format.
"""

import sqlite3

import pandas as pd
from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import Freq


def load_data_from_db() -> pd.DataFrame:
    """
    Extract the entire Freq table from the database.

    Returns:
        pd.DataFrame: DataFrame containing the Freq data.
    """
    logger.info('extracting the table from the database')
    query = select(Freq)
    return pd.read_sql(
        query,
        engine,
    )

