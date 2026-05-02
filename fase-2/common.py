"""Funciones compartidas de preprocesamiento para los scripts de la fase 2.

Este módulo centraliza el contrato de características para que entrenamiento
e inferencia apliquen siempre las mismas reglas de limpieza y generación de atributos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Lista ordenada de características usada tanto en entrenamiento como en predicción.
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

# Columnas crudas necesarias antes de generar atributos.
BASE_INPUT_COLS = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_datetime",
]

# Columna objetivo que aprende el modelo.
TRAIN_TARGET_COL = "trip_duration"
MODEL_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
PREDICTIONS_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "predictions" / "output.csv"


def haversine_distance(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Calcula la distancia Haversine en kilómetros entre dos pares de coordenadas."""
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
    """Lanza un error claro si al dataframe de entrada le faltan columnas requeridas."""
    required_set = set(required_columns)
    missing_columns = sorted(required_set - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def preprocess_dataframe(dataframe: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Limpia los datos crudos de taxis y devuelve la matriz de atributos usada por el modelo."""
    required_columns = list(BASE_INPUT_COLS)
    if require_target:
        required_columns.append(TRAIN_TARGET_COL)

    # Primero se valida que la entrada tenga los campos necesarios para esta fase.
    validate_required_columns(dataframe, required_columns)

    cleaned = dataframe.copy()
    # Elimina filas con valores nulos en las columnas que usa el modelo.
    cleaned = cleaned.dropna(subset=required_columns).copy()
    # Convierte la fecha de recogida a datetime para derivar variables temporales.
    cleaned["pickup_datetime"] = pd.to_datetime(cleaned["pickup_datetime"], errors="coerce")
    cleaned = cleaned.dropna(subset=["pickup_datetime"]).copy()

    # Filtra coordenadas obviamente inválidas y cantidades de pasajeros fuera de rango.
    nyc_box = (
        cleaned["pickup_longitude"].between(-74.3, -73.6)
        & cleaned["dropoff_longitude"].between(-74.3, -73.6)
        & cleaned["pickup_latitude"].between(40.5, 41.0)
        & cleaned["dropoff_latitude"].between(40.5, 41.0)
    )
    valid_passengers = cleaned["passenger_count"].between(1, 6)
    cleaned = cleaned[nyc_box & valid_passengers].copy()

    if require_target:
        # Durante el entrenamiento, conserva solo duraciones razonables del viaje.
        cleaned = cleaned[cleaned[TRAIN_TARGET_COL].between(60, 3600)].copy()

    # Generación de atributos compartida entre entrenamiento y predicción.
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
    """Extrae la matriz de atributos en el orden esperado por el modelo entrenado."""
    # Conserva exactamente el mismo orden de columnas con el que se entrenó el modelo.
    validate_required_columns(dataframe, FEATURE_COLS)
    return dataframe[FEATURE_COLS].copy()
