from src.predict import ModelUnavailableError
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import app as flask_app
from src.feature_extraction import FEATURE_NAMES, feature_vector


def test_pages_render():
    client = flask_app.app.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/about').status_code == 200
    assert client.get('/dataset').status_code == 200


def test_prediction_validates_url_before_loading_model(monkeypatch):
    def unexpected_load():
        raise AssertionError('model load must not happen for invalid input')
    monkeypatch.setattr(flask_app, 'load_bundle', unexpected_load)
    response = flask_app.app.test_client().post('/predict', data={'url': 'not a url'})
    assert response.status_code == 400


def test_missing_model_has_clear_message(monkeypatch):
    monkeypatch.setattr(flask_app, 'load_bundle', lambda: (_ for _ in ()).throw(ModelUnavailableError('The model has not been trained yet.')))
    response = flask_app.app.test_client().post('/predict', data={'url': 'https://example.com'})
    assert response.status_code == 503
    assert b'model has not been trained' in response.data


def test_prediction_route_renders_model_and_shap_result(monkeypatch):
    urls = [
        'https://safe.example.org/', 'https://news.example.net/article',
        'http://192.0.2.1/verify-account', 'http://secure-login.example.com/update?auth=1',
        'https://docs.example.edu/help', 'http://198.51.100.5/password-reset',
        'http://account-check.example.co/login', 'https://store.example.com/products',
    ]
    labels = [0, 0, 1, 1, 0, 1, 1, 0]
    matrix = pd.DataFrame([feature_vector(url) for url in urls], columns=FEATURE_NAMES)
    bundle = {
        'model': RandomForestClassifier(n_estimators=8, random_state=3).fit(matrix, labels),
        'feature_names': FEATURE_NAMES,
        'shap_background': matrix.iloc[:4].copy(),
    }
    monkeypatch.setattr(flask_app, 'load_bundle', lambda: bundle)
    response = flask_app.app.test_client().post('/predict', data={'url': 'http://192.0.2.1/verify'})
    assert response.status_code == 200
    assert b'URL ANALYSIS' in response.data
    assert b'class="shap-chart"' in response.data
    assert b'class="shap-bar' in response.data
    assert b'Extracted URL features' in response.data
