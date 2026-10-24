import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_extraction import FEATURE_NAMES, feature_vector
from src.predict import ModelUnavailableError, load_bundle, predict_url


def sample_bundle():
    urls = [
        "https://safe.example.org/", "https://news.example.net/article",
        "http://192.0.2.1/verify-account", "http://secure-login.example.com/update?auth=1",
        "https://docs.example.edu/help", "http://198.51.100.5/password-reset",
        "http://account-check.example.co/login", "https://store.example.com/products",
    ]
    labels = [0, 0, 1, 1, 0, 1, 1, 0]
    matrix = pd.DataFrame([feature_vector(url) for url in urls], columns=FEATURE_NAMES)
    model = RandomForestClassifier(n_estimators=8, random_state=3).fit(matrix, labels)
    return {"model": model, "feature_names": FEATURE_NAMES}


def test_prediction_returns_real_model_output_and_features():
    result = predict_url("http://192.0.2.1/verify", bundle=sample_bundle())
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["phishing_probability"] <= 1.0
    assert result["features"]["has_ip_address"] == 1
    assert result["label"] in {"Potentially phishing", "Likely legitimate"}


def test_missing_model_is_reported():
    with pytest.raises(ModelUnavailableError):
        load_bundle("models/definitely-missing.joblib")


def test_shap_explains_phishing_class():
    from src.explain import explain_prediction

    bundle = sample_bundle()
    bundle['shap_background'] = pd.DataFrame(
        [feature_vector(url) for url in ['https://safe.example.org/', 'http://192.0.2.1/verify-account']],
        columns=FEATURE_NAMES,
    )
    result = predict_url('http://192.0.2.1/verify', bundle=bundle)
    explanation = explain_prediction(bundle, result['features'], top_n=5)
    assert len(explanation['features']) == 5
    assert all('contribution' in feature and 'direction' in feature for feature in explanation['features'])
