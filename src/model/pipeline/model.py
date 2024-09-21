"""
This module creates the pipeline for building, training and saving ML model.

It includes the process of data preparation, model training using
RandomForestRegressor, hyperparameter tuning with GridSearchCV,
model evaluation, and serialization of the trained model.
"""

import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    """
    Build, evaluate and save a RandomForestRegressor model.

    This function orchestrates the model building pipeline.
    It starts by preparing the data, followed by defining deature names
    and splitting the dataset into features and target variables.
    The dataset is then divided into training and testing sets.
    The model's performance is evaluated on the test set, and
    finally, the model is saved for future use.

    Return:
        None
    """
    logger.info('starting up model building pipeline')
    df = prepare_data()
    feature_names = [
        'VehPower',
        'VehAge',
        'DrivAge',
        'BonusMalus',
        'Density',
        'Exposure',
        'Area',
        'VehBrand',
        'VehGas',
        'Region',
        ]
    response = 'Claim_Freq'
    X, y = _get_x_y(
        df,
        col_x=feature_names,
        col_y=response,
    )
    X_train, X_test, y_train, y_test = _split_train_test(
        X,
        y,
    )
    sample_weights_train = X_train['Exposure'].copy()
    sample_weights_test = X_test['Exposure'].copy()
    X_train = X_train.drop('Exposure', axis=1)
    X_test = X_test.drop('Exposure', axis=1)
    rf = _train_model(
        X_train,
        y_train,
        create_pipeline(),
        sample_weights_train,
    )
    _evaluate_model(
        rf,
        X_test,
        y_test,
        sample_weights_test,
    )
    _save_model(rf)


def _get_x_y(
    dataframe: pd.DataFrame,
    col_x: list[str],
    col_y: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into features and target variable.

    Args:
        dataframe (pd.DataFrame): The dataset to be split.
        col_x (list[str]): List of column names for features.
        col_y (str): Name of the target variable column.

    Returns:
        tuple: Features and target variables.
    """
    logger.info(f'defining X and Y variables. X vars: {col_x}; y var: {col_y}')
    return dataframe[col_x], dataframe[col_y]


def _split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        features (pd.DataFrame): Features dataset.
        target (pd.Series): Target variable.

    Returns:
        tuple: Training and testing sets for features and target.
    """
    logger.info('splitting data into train and test sets')
    return train_test_split(
        features,
        target,
        test_size=0.2,  # noqa: WPS432
    )


def create_pipeline():
    """
    Create the pipeline for the ML model.

    Returns:
        Pipeline: the pipeline for the ML model.
    """
    # Define the categorical features
    categorical_features = ['Area', 'VehBrand', 'VehGas', 'Region']

    # Create a ColumnTransformer to apply OneHotEncoder to categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
        ])

    # Return a pipeline that first preprocesses the data
    # and then applies a model
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor()),
    ])


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    sample_weights: pd.Series = None,
):
    """
    Train the RandomForestRegressor model with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training set features.
        y_train (pd.Series): Training set target.
        pipeline (Pipeline): The pipeline for the model.
        sample_weights (pd.Series): Sample weights for the observations.

    Returns:
        RandomForestRegressor: The best estimator after GridSearch.
    """
    logger.info('training a model with hyperparameters')

    # need the prefix to reference the step in the pipeline
    grid_space = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 6, 9, 12],
    }

    logger.debug(f'grid_space = {grid_space}')
    grid = GridSearchCV(
        pipeline,
        param_grid=grid_space,
        cv=5,
        scoring='r2',
    )

    fit_params = {'model__sample_weight': sample_weights}
    model_grid = grid.fit(
        X_train,
        y_train,
        # sample_weight=sample_weights,  # does not work?  # noqa: E800
        **fit_params,
    )
    return model_grid.best_estimator_


def _evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights,
) -> float:
    """
    Evaluate the trained model's performance.

    Args:
        model: The trained model or pipeline.
        X_test (pd.DataFrame): Testing set features.
        y_test (pd.Series): Testing set target.
        sample_weights (pd.Serpes): The weights of the test observations.

    Returns:
        float: The model's score.
    """
    model_score = model.score(
        X_test,
        y_test,
        sample_weights,
    )
    logger.info(f'evaluating model performance. SCORE={model_score}')
    return model_score


def _save_model(model: RandomForestRegressor) -> None:
    """
    Save the trained model to a specified directory.

    Args:
        model (RandomForestRegressor): The model to save.

    Return:
        None
    """
    model_path = f'{model_settings.model_path}/{model_settings.model_name}'
    logger.info(f'saving a model to a directory: {model_path}')
    with open(model_path, 'wb') as model_file:
        pk.dump(model, model_file)
