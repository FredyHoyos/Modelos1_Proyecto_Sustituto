"""Generate taxi duration predictions from a saved model and a CSV input file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import MODEL_DEFAULT_PATH, PREDICTIONS_DEFAULT_PATH, build_feature_matrix, preprocess_dataframe


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference."""
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
    """Load a serialized model from disk."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def prepare_input(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the phase 1 preprocessing steps to raw inference data."""
    prepared = preprocess_dataframe(input_frame, require_target=False)
    return build_feature_matrix(prepared)


def build_predictions_output(input_frame: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Create the output dataframe to be written as a CSV file."""
    output_frame = pd.DataFrame({"trip_duration": predictions})
    for identifier_column in ("id", "ID", "trip_id"):
        if identifier_column in input_frame.columns:
            output_frame.insert(0, identifier_column, input_frame.loc[predictions.index, identifier_column].to_numpy())
            break
    return output_frame


def save_predictions(predictions_frame: pd.DataFrame, output_path: Path) -> None:
    """Persist predictions to disk as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(output_path, index=False)


def main() -> None:
    """Run the prediction workflow from CLI arguments."""
    args = parse_args()
    input_path = Path(args.input_file)
    model_path = Path(args.model_file)
    output_path = Path(args.output_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = load_model(model_path)
    input_frame = pd.read_csv(input_path)
    feature_frame = prepare_input(input_frame)
    # Keep the prediction index aligned with the filtered feature rows so any preserved
    # identifier column matches the rows that survived preprocessing.
    predictions = pd.Series(model.predict(feature_frame), index=feature_frame.index, name="trip_duration")
    output_frame = build_predictions_output(input_frame, predictions)
    save_predictions(output_frame, output_path)

    print(f"Predicciones guardadas en: {output_path.resolve()}")


if __name__ == "__main__":
    main()
