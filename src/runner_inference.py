"""
Main application script for running the ML model service.

This script initializes the ModelInferenceService, loads the ML model,
makes a prediction based on predefined input parameters, and logs the output.
It demonstrates the typical workflow of using the ModelInferenceService in
a practical application context.
"""

import pandas as pd
from loguru import logger

from model.model_inference import ModelInferenceService


@logger.catch
def main():
    """
    Run the application.

    Load the model, make a prediction based on provided data,
    and log the prediction.
    """
    logger.info('running the application...')
    ml_svc = ModelInferenceService()
    ml_svc.load_model()

    # feature_values = {
    #     'area': 85,
    #     'constraction_year': 2015,
    #     'bedrooms': 2,
    #     'garden_area': 20,
    #     'balcony_present': 1,
    #     'parking_present': 1,
    #     'furnished': 0,
    #     'garage_present': 0,
    #     'storage_present': 1,
    # }
    feature_values = pd.DataFrame([{
        'VehPower': 5,
        'VehAge': 0,
        'DrivAge': 20,
        'BonusMalus': 100,
        'Exposure': 0.5,
        'Area': 'D',
        'VehBrand': 'B12',
        'VehGas': 'Regular',
        'Region': 'R82',
    }])
       
    # pred = ml_svc.predict(list(feature_values.values()))
    pred = ml_svc.predict(feature_values)
    logger.info(f'prediction = {pred}')


if __name__ == '__main__':
    main()
