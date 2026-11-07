"""
Train the Melbourne housing price model and save a deployment-ready pipeline.

Usage:
    python train_model.py

Outputs:
    model/housing_model.joblib
    model/model_metadata.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Melbourne_Housing.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "housing_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET = "Price"
CATEGORICAL_FEATURES = ["Suburb", "Type", "Regionname", "Postcode"]
NUMERIC_FEATURES = ["Rooms", "Distance", "Propertycount"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TYPE_LABELS = {"h": "House", "u": "Unit", "t": "Townhouse"}
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURES + [TARGET]].dropna().drop_duplicates()
    df["Postcode"] = df["Postcode"].astype(int).astype(str)
    df = df[df[TARGET] > 0]
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.06,
            max_leaf_nodes=63,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def build_metadata(df: pd.DataFrame, scores: dict, best_model: str, split: dict) -> dict:
    suburb_defaults = {}
    for suburb, group in df.groupby("Suburb", observed=True):
        suburb_defaults[str(suburb)] = {
            "postcode": str(group["Postcode"].mode().iat[0]),
            "region": str(group["Regionname"].mode().iat[0]),
            "propertycount": int(round(group["Propertycount"].median())),
            "distance": round(float(group["Distance"].median()), 1),
            "type": str(group["Type"].mode().iat[0]),
            "median_price": int(group[TARGET].median()),
        }

    best_metrics = scores[best_model]["test"]
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_rows": int(len(df)),
        "split": split,
        "best_model": best_model,
        "metrics": scores,
        "expected_error": round(best_metrics["mae"]),
        "price": {
            "median": int(df[TARGET].median()),
            "min": int(df[TARGET].min()),
            "max": int(df[TARGET].max()),
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        "type_labels": TYPE_LABELS,
        "suburbs": sorted(df["Suburb"].unique().tolist()),
        "postcodes": sorted(df["Postcode"].unique().tolist()),
        "regions": sorted(df["Regionname"].unique().tolist()),
        "suburb_defaults": suburb_defaults,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {DATA_PATH} ...")
    df = load_data()
    print(f"Rows: {len(df):,} | Columns used: {FEATURES}")

    X = df[FEATURES]
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    scores: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        print(f"\nTraining {name} ...")
        started = time.perf_counter()
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        train_scores = evaluate(y_train, train_pred)
        test_scores = evaluate(y_test, test_pred)
        scores[name] = {
            "train": train_scores,
            "test": test_scores,
            "fit_seconds": round(elapsed, 1),
        }
        fitted[name] = pipeline

        print(
            f"  train R2={train_scores['r2']:.4f} | "
            f"test R2={test_scores['r2']:.4f} | "
            f"test MAE=${test_scores['mae']:,.0f} | "
            f"test RMSE=${test_scores['rmse']:,.0f} | "
            f"{elapsed:.1f}s"
        )

    best_model = max(scores, key=lambda name: scores[name]["test"]["r2"])
    print(f"\nBest model: {best_model}")

    joblib.dump(fitted[best_model], MODEL_PATH)
    metadata = build_metadata(
        df,
        scores,
        best_model,
        split={
            "test_size": 0.2,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
        },
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model    -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
