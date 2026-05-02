"""Genera predicciones de duración de viajes desde un modelo guardado y un CSV de entrada.

El script replica el preprocesamiento del entrenamiento para que la inferencia
use la misma estructura de atributos que el modelo ajustado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import MODEL_DEFAULT_PATH, PREDICTIONS_DEFAULT_PATH, build_feature_matrix, preprocess_dataframe


def parse_args() -> argparse.Namespace:
    """Analiza los argumentos de línea de comandos para la inferencia."""
    parser = argparse.ArgumentParser(description="Predict taxi trip duration from a CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--model_file",
        type=str,
        default=str(MODEL_DEFAULT_PATH),
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(PREDICTIONS_DEFAULT_PATH),
        help="Path where the predictions CSV will be saved.",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    """Carga un modelo serializado desde disco."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def prepare_input(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica al dato crudo de inferencia los mismos pasos de preprocesamiento de la fase 1."""
    # Reutiliza exactamente la misma generación de atributos usada en entrenamiento.
    prepared = preprocess_dataframe(input_frame, require_target=False)
    return build_feature_matrix(prepared)


def build_predictions_output(input_frame: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Crea el dataframe de salida que se escribirá como CSV."""
    output_frame = pd.DataFrame({"trip_duration": predictions})
    # Conserva una columna identificadora útil cuando el archivo de entrada la trae.
    for identifier_column in ("id", "ID", "trip_id"):
        if identifier_column in input_frame.columns:
            output_frame.insert(0, identifier_column, input_frame.loc[predictions.index, identifier_column].to_numpy())
            break
    return output_frame


def save_predictions(predictions_frame: pd.DataFrame, output_path: Path) -> None:
    """Guarda las predicciones en disco como archivo CSV."""
    # Asegura que la carpeta de salida exista antes de escribir el CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(output_path, index=False)


def main() -> None:
    """Ejecuta el flujo de predicción a partir de los argumentos CLI."""
    args = parse_args()
    input_path = Path(args.input_file)
    model_path = Path(args.model_file)
    output_path = Path(args.output_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = load_model(model_path)
    input_frame = pd.read_csv(input_path)
    feature_frame = prepare_input(input_frame)
    # Mantiene alineado el índice de predicción con las filas filtradas para que
    # cualquier columna identificadora preservada coincida con las filas que sobrevivieron.
    predictions = pd.Series(model.predict(feature_frame), index=feature_frame.index, name="trip_duration")
    output_frame = build_predictions_output(input_frame, predictions)
    save_predictions(output_frame, output_path)

    print(f"Predicciones guardadas en: {output_path.resolve()}")


if __name__ == "__main__":
    main()
