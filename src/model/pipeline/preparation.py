"""
This module provides functionality for preparing a dataset for ML model.

It consists of functions to load data from a database, column encodings,
and parse specific columns for further processing.
Some encodings may be handled by the model pipelines directly in model.py.
"""


import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db


def prepare_data() -> pd.DataFrame:
    """
    Prepare the dataset for analysis and modelling.

    This involves loading the data, and encoding the response variable.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    logger.info('starting up preprocessing pipeline')
    dataframe = load_data_from_db()

    return _encode_response(dataframe)


def _encode_response(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['Claim_Freq'] = dataframe['ClaimNb']/dataframe['Exposure']
    return dataframe
