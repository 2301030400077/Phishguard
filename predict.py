"""Model loading and inference helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.feature_extraction import FEATURE_NAMES, extract_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "phishguard_model.joblib"


class ModelUnavailableError(RuntimeError):
    """Raised when no compatible trained model is available."""


@lru_cache(maxsize=2)
def load_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise ModelUnavailableError("The model has not been trained yet. Follow the README training steps.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelUnavailableError("The saved model could not be loaded. Retrain it with the current pipeline.") from exc
    if not isinstance(bundle, dict) or "model" not in bundle or bundle.get("feature_names") != FEATURE_NAMES:
        raise ModelUnavailableError("Saved model is incompatible. Retrain it with the current feature pipeline.")
    # Inference on one URL is small; a single worker avoids joblib process/pipe overhead.
    if hasattr(bundle["model"], "n_jobs"):
        bundle["model"].n_jobs = 1
    if "vectorizer" in bundle and bundle.get("vectorizer") is None:
        raise ModelUnavailableError("The saved URL text vectorizer is missing. Retrain the model.")
    return bundle


def predict_url(url: str, bundle: dict | None = None, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    bundle = bundle if bundle is not None else load_bundle(model_path)
    features = extract_features(url)
    model = bundle["model"]
    if bundle.get("vectorizer") is not None:
        row = bundle["vectorizer"].transform([url])
    else:
        row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
    probability = None
    if probabilities is not None:
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
    return {
        "prediction": prediction,
        "label": "Potentially phishing" if prediction == 1 else "Likely legitimate",
        "phishing_probability": probability,
        "features": features,
    }
