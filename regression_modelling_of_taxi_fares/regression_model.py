#!/usr/bin/env python
# coding: utf-8
"""
End-to-end workflow for preparing NYC Taxi data, engineering features,
training a linear regression model, evaluating performance, generating
diagnostic plots, and producing full-dataset predictions.
"""

import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# Logging configuration
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_FILE = "2017_Yellow_Taxi_Trip_Data.csv"
IQR_FACTOR = 6
TEST_SIZE = 0.20
RANDOM_STATE = 0


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    logging.info("Loading dataset")
    df = pd.read_csv(path)
    return df.copy()


def convert_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pickup and dropoff timestamps to datetime."""
    df['tpep_pickup_datetime'] = pd.to_datetime(
        df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p'
    )
    df['tpep_dropoff_datetime'] = pd.to_datetime(
        df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p'
    )
    return df


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Create trip duration column in minutes."""
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']) \
        / np.timedelta64(1, 'm')
    return df


def outlier_imputer(df: pd.DataFrame, column_list, iqr_factor: float):
    """Apply IQR-based upper-limit capping and force negative values to zero."""
    for col in column_list:
        df.loc[df[col] < 0, col] = 0

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + (iqr_factor * iqr)

        df.loc[df[col] > upper, col] = upper


def prepare_mean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean distance and duration per pickup/dropoff pair."""
    df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)

    dist_dict = df.groupby('pickup_dropoff').mean(numeric_only=True)['trip_distance'].to_dict()
    dur_dict = df.groupby('pickup_dropoff').mean(numeric_only=True)['duration'].to_dict()

    df['mean_distance'] = df['pickup_dropoff'].map(dist_dict)
    df['mean_duration'] = df['pickup_dropoff'].map(dur_dict)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day name, month, and rush-hour indicator."""
    df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()
    df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
    df['hour'] = df['tpep_pickup_datetime'].dt.hour

    def rush_hour(h, d):
        if d in ('saturday', 'sunday'):
            return 0
        if 6 <= h < 10 or 16 <= h < 20:
            return 1
        return 0

    df['rush_hour'] = df.apply(lambda r: rush_hour(r['hour'], r['day']), axis=1)
    return df


def isolate_modeling_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not used as predictors or not available at prediction time."""
    drop_cols = [
        'Unnamed: 0', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID',
        'DOLocationID', 'payment_type', 'extra', 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge', 'total_amount',
        'duration', 'pickup_dropoff', 'day', 'month', 'hour'
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def prepare_training_data(df2: pd.DataFrame):
    """Split, scale, and encode training and test sets."""
    X = df2.drop(columns=['fare_amount'])
    y = df2[['fare_amount']]

    X['VendorID'] = X['VendorID'].astype(str)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled, y_train, y_test, X, X_train


def train_model(X_train_scaled, y_train):
    """Fit linear regression model."""
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model


def evaluate(model, X_scaled, y_true):
    """Generate predictions and evaluation metrics."""
    preds = model.predict(X_scaled)
    metrics = {
        'r2': r2_score(y_true, preds),
        'mae': mean_absolute_error(y_true, preds),
        'mse': mean_squared_error(y_true, preds),
        'rmse': np.sqrt(mean_squared_error(y_true, preds)),
    }
    return preds, metrics


def build_results_df(y_test, y_pred):
    """Create dataframe of actual vs predicted values."""
    df = pd.DataFrame({
        'actual': y_test['fare_amount'],
        'predicted': y_pred.ravel()
    })
    df['residual'] = df['actual'] - df['predicted']
    return df


def plot_diagnostics(results: pd.DataFrame):
    """Generate model diagnostic plots."""
    sns.set(style='whitegrid')

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='actual', y='predicted', data=results, s=20, alpha=0.5)
    plt.plot([0, 60], [0, 60], c='red')
    plt.title('Actual vs Predicted')
    plt.show()

    plt.figure()
    sns.histplot(results['residual'], bins=np.arange(-15, 16, 0.5))
    plt.title('Residual Distribution')
    plt.show()

    plt.figure()
    sns.scatterplot(x='predicted', y='residual', data=results)
    plt.axhline(0, c='red')
    plt.title('Residuals vs Predicted')
    plt.show()


def predict_full(df, scaler, model, X):
    """Predict the entire dataset and adjust fare for RatecodeID == 2."""
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled).ravel()

    adjusted = df[['RatecodeID']].copy()
    adjusted['pred'] = preds
    adjusted.loc[adjusted['RatecodeID'] == 2, 'pred'] = 52

    result = df[['mean_duration', 'mean_distance']].copy()
    result['predicted_fare'] = adjusted['pred'].values
    return adjusted['pred'], result


def main():
    df = load_data(DATA_FILE)

    logging.info("Initial shape: %s", df.shape)

    df = convert_datetimes(df)
    df = add_duration(df)

    df.loc[df['fare_amount'] < 0, 'fare_amount'] = 0
    df.loc[df['duration'] < 0, 'duration'] = 0

    outlier_imputer(df, ['fare_amount', 'duration'], IQR_FACTOR)

    df = prepare_mean_features(df)
    df = add_time_features(df)

    df2 = isolate_modeling_variables(df)

    scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_full, X_train = \
        prepare_training_data(df2)

    model = train_model(X_train_scaled, y_train)

    y_pred_train, train_metrics = evaluate(model, X_train_scaled, y_train)
    y_pred_test, test_metrics = evaluate(model, X_test_scaled, y_test)

    logging.info("Train metrics: %s", train_metrics)
    logging.info("Test metrics: %s", test_metrics)

    results = build_results_df(y_test, y_pred_test)
    plot_diagnostics(results)

    final_preds, final_df = predict_full(df, scaler, model, X_full)

    logging.info("Completed processing.")


if __name__ == "__main__":
    main()
