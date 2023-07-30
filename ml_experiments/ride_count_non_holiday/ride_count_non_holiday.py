"""
This file contains the featurization opportunities for the ride count on non-holiday.
"""

# OS utilities
import os

# System
import sys

# Prefect tasks.
from prefect import flow, task

sys.path.insert(0, "../../")
import math

# Graphing
import matplotlib.pyplot as plt

# MLOps utilities
import mlflow  # general logging
import numpy as np

# Data manipulation
import pandas as pd
from sklearn.cluster import KMeans

# Training utilities for machine learning
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
)

# machine learning utilities
from sklearn.linear_model import LinearRegression, LogisticRegression

# Metrics for machine learning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Utilities import
from utilities import *

# Set up the tracking uri - should just be default
mlflow.set_tracking_uri(mlflow_tracking_uri)


@task(name="Featurization for Ride Count on Non-Holidays")
def featurize_ride_count_non_holiday():
    """
    Featurize for ride count on non-holidays
    """
    # Load in the desired dataset.
    gold_df = pd.read_parquet(
        data_directory_format.format(
            DataQualityEnum.GOLD, data_pipeline_file_format.format(DataQualityEnum.GOLD)
        )
    )
    # Remove the holidays from the feature set.
    gold_df = gold_df[gold_df["holiday"] == 0]
    # Aggregate the data to the day level.
    count_feature_df = gold_df[["start_date", "bikeid"]].groupby(["start_date"]).count()
    # Aggregate the mean information for the other items.
    median_feature_df = (
        gold_df[["start_date", "duration_minutes", "month", "day_of_week", "season"]]
        .groupby(["start_date"])
        .mean()
        .astype(int)
    )
    # Merge back together.
    merge_df = count_feature_df.merge(median_feature_df, how="inner", on="start_date")
    # rename the count
    merge_df.rename({"bikeid": "ride_count"}, axis=1, inplace=True)
    # featurize the weekday vs weekend back into the dataset.
    merge_df["weekend"] = merge_df["day_of_week"].apply(
        lambda x: 0.0 if x < 5.0 else 1.0
    )
    # Save the base data to the featurized data section.
    merge_df.to_parquet(
        experiment_data_format.format(ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "features")
    )
    # Return the dataframe.
    return merge_df


@task(name="Featurization per Day for Ride Count on Non-Holidays")
def featurize_per_week_ride_count_non_holiday(features_df: pd.DataFrame):
    """
    This method featurizes the data by the previous week of transactions
    """
    # Copy the dataframe
    features_per_day_df = features_df.copy()
    # Apply rolling window features of 7 days
    features_per_day_df["past_week_total"] = (
        features_per_day_df["ride_count"]
        .rolling(8)
        .apply(lambda x: sum(x.iloc[: len(x) - 1]))
    )
    features_per_day_df["past_week_mean"] = (
        features_per_day_df["ride_count"]
        .rolling(8)
        .apply(lambda x: sum(x.iloc[: len(x) - 1]) / (len(x) - 1))
    )
    # Save the data
    features_per_day_df.dropna(axis=0).to_parquet(
        experiment_data_format.format(
            ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "features_per_day"
        )
    )
    # Return the dataframe
    return features_per_day_df


@task(name="Featurization per Month for Ride Count on Non-Holidays")
def featurize_per_month_ride_count_non_holiday(features_df: pd.DataFrame):
    """
    This method featurizes the data by the previous month of transactions.
    """
    # Copy the dataframe
    features_per_month_df = features_df.copy()
    # Apply rolling window features of 30 days
    features_per_month_df["past_month_total"] = (
        features_per_month_df["ride_count"]
        .rolling(31)
        .apply(lambda x: sum(x.iloc[: len(x) - 1]))
    )
    features_per_month_df["past_month_mean"] = (
        features_per_month_df["ride_count"]
        .rolling(31)
        .apply(lambda x: sum(x.iloc[: len(x) - 1]) / (len(x) - 1))
    )
    # Save the data
    features_per_month_df.dropna(axis=0).to_parquet(
        experiment_data_format.format(
            ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "features_per_month"
        )
    )
    # Return the data
    return features_per_month_df


@task(name="ML Experiment: Ride Count Non Holiday")
def ml_experiment_ride_count_non_holiday(features_df: pd.DataFrame):
    """
    Test the data against the linear regression algorithm.
    Data Schema: ride_count, duration_minutes, month, day_of_week, season, weekend, past_week_total, past_week_mean, past_month_total, past_month_mean
    """
    # Experiment dataframe
    experiment_df = features_df.copy().dropna(axis=0)
    # Feature and label columns
    category_columns = ["month", "day_of_week", "season", "weekend"]
    label_column = "ride_count"
    # Convert the category columns
    for cat_col in category_columns:
        # Merge in the columns
        experiment_df = pd.concat(
            (experiment_df, pd.get_dummies(experiment_df[cat_col])), axis=1
        )
        # Rename the columns
        experiment_df.columns = [
            col if type(col) == str else f"{cat_col}_{col}"
            for col in experiment_df.columns
        ]
    # Drop the category columns
    experiment_df.drop(category_columns, axis=1, inplace=True)
    # Save the dataset
    experiment_df.to_parquet(
        experiment_data_format.format(ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "ml_input")
    )
    # Test train and split on the base features. Stratify on the day of the week.
    x_train, x_test, y_train, y_test = train_test_split(
        experiment_df[[col for col in experiment_df.columns if col != label_column]],
        experiment_df[label_column],
        test_size=0.3,
    )
    # Run the default linear regressor ride count non holidays
    ml_experiment_linear_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_logistic_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_decision_tree_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_random_forest_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_kmeans_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_xgboost_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    # Run the bagged classifiers
    ml_experiment_bagged_linear_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_bagged_logistic_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_bagged_decision_tree_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_bagged_random_forest_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_bagged_kmeans_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_bagged_xgboost_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    # Run the boosted classifiers
    ml_experiment_boosted_linear_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_boosted_logistic_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_boosted_decision_tree_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_boosted_random_forest_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_boosted_kmeans_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )
    ml_experiment_boosted_xgboost_regressor_ride_count_non_holidays(
        x_train, x_test, y_train, y_test
    )


def ml_experiment_linear_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        lr = LinearRegression()  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "linear_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "linear_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", lr.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_default_linear_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_default_linear_regression"
            )
        )


def ml_experiment_logistic_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        lr = LogisticRegression()  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "logistic_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "logistic_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", lr.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_default_logistic_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_default_logistic_regression",
            )
        )


def ml_experiment_kmeans_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        lr = KMeans()  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "kmeans_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "kmeans_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", lr.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_default_kmeans_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_default_kmeans_regression"
            )
        )


def ml_experiment_decision_tree_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        lr = DecisionTreeRegressor()  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "decision_tree_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "decision_tree_regression_default_testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", lr.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_default_decision_tree_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_default_decision_tree_regression",
            )
        )


def ml_experiment_random_forest_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        lr = RandomForestRegressor()  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "random_forest_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "random_forest_regression_default_testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", lr.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_default_random_forest_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_default_random_forest_regression",
            )
        )


def ml_experiment_xgboost_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        lr = XGBRegressor()  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.xgboost.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "xgboost_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "xgboost_regression_default_testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", lr.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_default_xgboost_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_default_xgboost_regression"
            )
        )


# ===============================================================
# Start of the bagged utilities
# ===============================================================


def ml_experiment_bagged_linear_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = LinearRegression()
        lr = BaggingRegressor(estimator=m, bootstrap=True)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_linear_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_linear_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", True)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_bagged_linear_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_bagged_linear_regression"
            )
        )


def ml_experiment_bagged_logistic_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = LogisticRegression()
        lr = BaggingRegressor(m, bootstrap=True)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_logistic_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_logistic_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", True)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_bagged_logistic_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_bagged_logistic_regression"
            )
        )


def ml_experiment_bagged_kmeans_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = KMeans()
        lr = BaggingRegressor(estimator=m, bootstrap=True)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_kmeans_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_kmeans_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", True)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_bagged_kmeans_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_bagged_kmeans_regression"
            )
        )


def ml_experiment_bagged_decision_tree_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = DecisionTreeRegressor()
        lr = BaggingRegressor(estimator=m, bootstrap=True)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "bagged_decision_tree_regression_default_train_error.png"
        )
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "bagged_decision_tree_regression_default_testing_error.png"
        )
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", True)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_bagged_decision_tree_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_bagged_decison_tree_regression",
            )
        )


def ml_experiment_bagged_random_forest_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = RandomForestRegressor()
        lr = BaggingRegressor(m, bootstrap=True)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "bagged_random_forest_regression_default_train_error.png"
        )
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "bagged_random_forest_regression_default_testing_error.png"
        )
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", True)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_bagged_random_forest_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_bagged_random_forest_regression",
            )
        )


def ml_experiment_bagged_xgboost_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = XGBRegressor()
        lr = BaggingRegressor(estimator=m, bootstrap=True)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_xgboost_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "bagged_xgboost_regression_default_testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", True)
        mlflow.log_param("boosted", False)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_bagged_xgboost_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_bagged_xgboost_regression"
            )
        )


# ===============================================================
# Start of the boosted utilities
# ===============================================================


def ml_experiment_boosted_linear_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = LinearRegression()
        lr = AdaBoostRegressor(estimator=m)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_linear_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_linear_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", True)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_boosted_linear_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_boosted_linear_regression"
            )
        )


def ml_experiment_boosted_logistic_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = LogisticRegression()
        lr = AdaBoostRegressor(m)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_logistic_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_logistic_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", True)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_boosted_logistic_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_boosted_logistic_regression",
            )
        )


def ml_experiment_boosted_kmeans_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = KMeans()
        lr = AdaBoostRegressor(estimator=m)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_kmeans_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_kmeans_regression_default_Testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", True)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "train_boosted_kmeans_regression"
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY, "test_boosted_kmeans_regression"
            )
        )


def ml_experiment_boosted_decision_tree_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = DecisionTreeRegressor()
        lr = AdaBoostRegressor(estimator=m)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "boosted_decision_tree_regression_default_train_error.png"
        )
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "boosted_decision_tree_regression_default_testing_error.png"
        )
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", True)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_boosted_decision_tree_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_boosted_decison_tree_regression",
            )
        )


def ml_experiment_boosted_random_forest_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = RandomForestRegressor()
        lr = AdaBoostRegressor(m)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "boosted_random_forest_regression_default_train_error.png"
        )
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(
            fig, "boosted_random_forest_regression_default_testing_error.png"
        )
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", True)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_boosted_random_forest_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_boosted_random_forest_regression",
            )
        )


def ml_experiment_boosted_xgboost_regressor_ride_count_non_holidays(
    x_train, x_test, y_train, y_test
):
    """
    Use a Linear Regression default experiment to predict the ride count on non-holidays
    """
    # Apply the Linear Regression for a test.
    with mlflow.start_run(experiment_id=1):
        # Train the model
        m = XGBRegressor()
        lr = AdaBoostRegressor(estimator=m)  # Default parameters
        # Fit the model
        lr.fit(x_train, y_train)
        # Test the model on training data
        train_preds = lr.predict(x_train)
        # Test the model on testing data
        test_preds = lr.predict(x_test)
        # Calculate the Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        # Calculate the Mean Squared Error
        train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
        # Log the model for sklearn
        mlflow.sklearn.log_model(lr, artifact_path="model")
        # Log the metrics
        mlflow.log_metric("Training Mean Absolute Error", train_mae)
        mlflow.log_metric("Testing Mean Absolute Error", test_mae)
        mlflow.log_metric("Training RMSE", train_rmse)
        mlflow.log_metric("Testing RMSE", test_rmse)
        # Log artifact graph of training predictions
        fig, ax = plt.subplots()
        train_series_mae = y_train - train_preds
        plt.scatter(x=x_train.index, y=train_series_mae)
        plt.title("MAE of Training Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_xgboost_regression_default_train_error.png")
        # log MAE artifact graph of testing predictions
        fig, ax = plt.subplots()
        test_series_mae = y_test - test_preds
        plt.scatter(x=x_test.index, y=test_series_mae)
        plt.title("MAE of Testing Predictions")
        plt.xlabel("Data Point")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        mlflow.log_figure(fig, "boosted_xgboost_regression_default_testing_error.png")
        # Log the tags
        mlflow.log_param("production", False)
        mlflow.log_param("bagged", False)
        mlflow.log_param("boosted", True)
        mlflow.log_param("model_type", m.__class__)
        # Assemble output datasets
        train_df = pd.concat((x_train, y_train), axis=1)
        train_df["predictions"] = train_preds
        test_df = pd.concat((x_test, y_test), axis=1)
        test_df["predictions"] = test_preds
        # Save datasets to folder structure.
        train_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "train_boosted_random_forest_regression",
            )
        )
        test_df.to_parquet(
            experiment_data_format.format(
                ExperimentEnum.RIDE_COUNT_NON_HOLIDAY,
                "test_boosted_random_forest_regression",
            )
        )
