"""Train the taxi duration model and persist it to disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import MODEL_DEFAULT_PATH, TRAIN_TARGET_COL, build_feature_matrix, preprocess_dataframe


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training."""
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
    """Fit the regression model using the phase 1 feature contract."""
    prepared = preprocess_dataframe(training_frame, require_target=True)
    features = build_feature_matrix(prepared)
    target = prepared[TRAIN_TARGET_COL]

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
    """Persist the fitted model to the requested path."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def main() -> None:
    """Run the training workflow from CLI arguments."""
    args = parse_args()
    input_path = Path(args.input_file)
    model_path = Path(args.model_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Training file not found: {input_path}")

    training_frame = pd.read_csv(input_path)
    model = train_model(training_frame)
    save_model(model, model_path)

    print(f"Modelo guardado en: {model_path.resolve()}")


if __name__ == "__main__":
    main()
