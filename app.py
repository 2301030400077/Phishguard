import json
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, render_template, request

from src.explain import explain_prediction
from src.predict import ModelUnavailableError, load_bundle, predict_url

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024
app.config['MAX_URL_LENGTH'] = 2048
ROOT = Path(__file__).resolve().parent


def valid_url_input(value: str) -> bool:
    if not value or len(value) > app.config['MAX_URL_LENGTH'] or any(char.isspace() for char in value):
        return False
    candidate = value if '://' in value else '//' + value
    try:
        parts = urlsplit(candidate)
        return parts.scheme.lower() in {'', 'http', 'https'} and bool(parts.hostname) and (parts.port is None or 0 < parts.port < 65536)
    except ValueError:
        return False


@app.get('/')
def home():
    return render_template('index.html')


@app.post('/predict')
def predict():
    submitted_url = request.form.get('url', '').strip()
    if not valid_url_input(submitted_url):
        return render_template('index.html', error='Enter a valid HTTP or HTTPS URL (or hostname), up to 2,048 characters.'), 400
    try:
        bundle = load_bundle()
    except ModelUnavailableError as exc:
        return render_template('result.html', submitted_url=submitted_url,
                               unavailable=True, model_error=str(exc)), 503
    try:
        result = predict_url(submitted_url, bundle=bundle)
    except Exception:
        app.logger.exception('URL prediction failed')
        return render_template('result.html', submitted_url=submitted_url, prediction_error=True), 500
    try:
        explanation = explain_prediction(bundle, result['features'], url=submitted_url)
        explanation_error = None
    except Exception:
        app.logger.exception('SHAP explanation failed')
        explanation = None
        explanation_error = 'The model prediction is available, but a SHAP explanation could not be generated.'
    return render_template('result.html', submitted_url=submitted_url, result=result,
                           explanation=explanation, explanation_error=explanation_error)


@app.get('/about')
def about():
    return render_template('about.html')


@app.get('/dataset')
def dataset():
    metrics_path = ROOT / 'models' / 'metrics.json'
    metrics = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            app.logger.exception('Could not read model metrics')
    return render_template('dataset.html', metrics=metrics)


@app.errorhandler(413)
def request_too_large(_error):
    return render_template('index.html', error='Request is too large. URL must be 2,048 characters or fewer.'), 413


if __name__ == '__main__':
    app.run(debug=False)
