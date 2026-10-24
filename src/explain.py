"""SHAP explanations for the phishing class (label 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.feature_extraction import FEATURE_NAMES


def explain_prediction(bundle: dict, features: dict[str, int], top_n: int = 8, url: str | None = None) -> dict:
    """Explain contributions to phishing-class probability; fail clearly if SHAP fails."""
    if bundle.get("vectorizer") is not None and url is not None:
        model = bundle["model"]
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("The fitted model does not contain the phishing class (1).")
        row = bundle["vectorizer"].transform([url])
        active = row.tocoo()
        if len(classes) == 2 and model.coef_.shape[0] == 1:
            class_sign = 1.0 if classes.index(1) == 1 else -1.0
            weights = model.coef_[0] * class_sign
            base_value = float(model.intercept_[0] * class_sign)
        else:
            weights = model.coef_[classes.index(1)]
            base_value = float(model.intercept_[classes.index(1)])
        pattern_names = bundle["vectorizer"].get_feature_names_out()
        ranked = sorted(
            ({"name": f"URL pattern: {pattern_names[index]}",
              "value": round(float(value), 3), "contribution": float(value * weights[index])}
             for index, value in zip(active.col, active.data)),
            key=lambda item: abs(item["contribution"]), reverse=True,
        )[:top_n]
        for item in ranked:
            item["direction"] = "increases" if item["contribution"] > 0 else "decreases" if item["contribution"] < 0 else "does not change"
        return {"base_value": base_value, "features": ranked}

    model = bundle.get("explanation_model", bundle["model"])
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain the phishing class (1).")
    phishing_index = classes.index(1)
    row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    background = bundle.get("shap_background")
    if background is None or len(background) == 0:
        raise ValueError("SHAP background data is missing; retrain the model.")
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional", model_output="probability"
    )
    explanation = explainer(row)
    raw_values = explanation.values
    if isinstance(raw_values, list):
        values = np.stack(raw_values, axis=-1)
    else:
        values = np.asarray(raw_values)
    if values.ndim == 3:
        contributions = values[0, :, phishing_index]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError("SHAP returned an unsupported explanation shape.")
    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 2:
        base_value = float(base_values[0, phishing_index])
    elif base_values.ndim == 1 and len(base_values) == len(classes):
        base_value = float(base_values[phishing_index])
    else:
        base_value = float(base_values.reshape(-1)[0])
    ranked = sorted(
        ({"name": name, "value": int(features[name]), "contribution": float(contributions[i])}
         for i, name in enumerate(FEATURE_NAMES)),
        key=lambda item: abs(item["contribution"]), reverse=True,
    )[:top_n]
    for item in ranked:
        item["direction"] = "increases" if item["contribution"] > 0 else "decreases" if item["contribution"] < 0 else "does not change"
    return {"base_value": base_value, "features": ranked}
