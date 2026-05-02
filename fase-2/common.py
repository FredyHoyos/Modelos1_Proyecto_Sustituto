"""Shared preprocessing helpers for phase 2 scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_distance",
    "delta_lat",
    "delta_lon",
    "trip_distance_sq",
    "hour",
    "day_of_week",
    "is_weekend",
]

BASE_INPUT_COLS = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_datetime",
]

TRAIN_TARGET_COL = "trip_duration"
MODEL_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
PREDICTIONS_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "predictions" / "output.csv"


def haversine_distance(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Compute the Haversine distance in kilometers between two coordinate pairs."""
    radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius_km * c


def validate_required_columns(dataframe: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if the input dataframe misses any required columns."""
    required_set = set(required_columns)
    missing_columns = sorted(required_set - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def preprocess_dataframe(dataframe: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Clean raw taxi data and return the feature matrix used by the model."""
    required_columns = list(BASE_INPUT_COLS)
    if require_target:
        required_columns.append(TRAIN_TARGET_COL)

    validate_required_columns(dataframe, required_columns)

    cleaned = dataframe.copy()
    cleaned = cleaned.dropna(subset=required_columns).copy()
    cleaned["pickup_datetime"] = pd.to_datetime(cleaned["pickup_datetime"], errors="coerce")
    cleaned = cleaned.dropna(subset=["pickup_datetime"]).copy()

    nyc_box = (
        cleaned["pickup_longitude"].between(-74.3, -73.6)
        & cleaned["dropoff_longitude"].between(-74.3, -73.6)
        & cleaned["pickup_latitude"].between(40.5, 41.0)
        & cleaned["dropoff_latitude"].between(40.5, 41.0)
    )
    valid_passengers = cleaned["passenger_count"].between(1, 6)
    cleaned = cleaned[nyc_box & valid_passengers].copy()

    if require_target:
        cleaned = cleaned[cleaned[TRAIN_TARGET_COL].between(60, 3600)].copy()

    cleaned["trip_distance"] = haversine_distance(
        cleaned["pickup_latitude"],
        cleaned["pickup_longitude"],
        cleaned["dropoff_latitude"],
        cleaned["dropoff_longitude"],
    )
    cleaned["delta_lat"] = cleaned["dropoff_latitude"] - cleaned["pickup_latitude"]
    cleaned["delta_lon"] = cleaned["dropoff_longitude"] - cleaned["pickup_longitude"]
    cleaned["trip_distance_sq"] = cleaned["trip_distance"] ** 2
    cleaned["hour"] = cleaned["pickup_datetime"].dt.hour
    cleaned["day_of_week"] = cleaned["pickup_datetime"].dt.dayofweek
    cleaned["is_weekend"] = (cleaned["day_of_week"] >= 5).astype(int)

    return cleaned


def build_feature_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract the ordered feature matrix expected by the trained model."""
    validate_required_columns(dataframe, FEATURE_COLS)
    return dataframe[FEATURE_COLS].copy()
