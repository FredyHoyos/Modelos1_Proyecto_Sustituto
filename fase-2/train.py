"""Entrena el modelo de duración de viajes y lo guarda en disco.

El script lee un CSV, aplica el preprocesamiento compartido,
entrena un regresor de bosque aleatorio y guarda el modelo ajustado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import MODEL_DEFAULT_PATH, TRAIN_TARGET_COL, build_feature_matrix, preprocess_dataframe


def parse_args() -> argparse.Namespace:
    """Analiza los argumentos de línea de comandos para el entrenamiento."""
    parser = argparse.ArgumentParser(description="Train a taxi duration model from a CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument(
        "--model_file",
        type=str,
        default=str(MODEL_DEFAULT_PATH),
        help="Path where the trained model will be saved.",
    )
    return parser.parse_args()


def train_model(training_frame: pd.DataFrame) -> RandomForestRegressor:
    """Ajusta el modelo de regresión usando el contrato de atributos de la fase 1."""
    prepared = preprocess_dataframe(training_frame, require_target=True)
    features = build_feature_matrix(prepared)
    target = prepared[TRAIN_TARGET_COL]

    # El bosque aleatorio es una base sólida para este problema tabular de regresión.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features, target)
    return model


def save_model(model: RandomForestRegressor, model_path: Path) -> None:
    """Guarda el modelo ajustado en la ruta solicitada."""
    # Crea la carpeta destino si todavía no existe.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def main() -> None:
    """Ejecuta el flujo de entrenamiento a partir de los argumentos CLI."""
    args = parse_args()
    input_path = Path(args.input_file)
    model_path = Path(args.model_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Training file not found: {input_path}")

    # Carga los datos crudos, ajusta el modelo y lo guarda en disco.
    training_frame = pd.read_csv(input_path)
    model = train_model(training_frame)
    save_model(model, model_path)

    print(f"Modelo guardado en: {model_path.resolve()}")


if __name__ == "__main__":
    main()
