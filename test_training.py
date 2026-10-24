import pandas as pd

from src.explain import explain_prediction
from src.predict import load_bundle, predict_url
from src.train_model import load_dataset, train


def test_dataset_cleaning_uses_only_url_and_label(tmp_path):
    rows = []
    for index in range(12):
        label = index % 2  # PhiUSIIL encoding: 0 phishing, 1 legitimate.
        rows.append({
            'URL': f"http://{'bad' if label == 0 else 'good'}{index}.example{index}.com/login",
            'label': label,
            'URLLength': 5000 + index,  # Must never enter the model feature matrix.
        })
    rows.extend([
        {'URL': rows[0]['URL'], 'label': rows[0]['label'], 'URLLength': -1},
        {'URL': None, 'label': 0, 'URLLength': 0},
        {'URL': 'not a URL', 'label': 1, 'URLLength': 3},
        {'URL': 'https://missing-label.example.com/', 'label': None, 'URLLength': 0},
    ])
    dataset = tmp_path / 'sample.csv'
    pd.DataFrame(rows).to_csv(dataset, index=False)
    features, labels, metadata = load_dataset(dataset)
    assert list(features.columns) == [
        'url_length', 'hostname_length', 'dot_count', 'hyphen_count', 'digit_count',
        'special_char_count', 'subdomain_count', 'has_ip_address', 'has_at_symbol',
        'uses_https', 'suspicious_keyword_count', 'path_length', 'query_length', 'parameter_count',
    ]
    assert len(features) == 12
    assert labels.sum() == 6  # UCI phishing label 0 mapped to PhishGuard's class 1.
    assert metadata['removed_rows'] == 4


def test_end_to_end_training_saves_metrics_prediction_and_shap(tmp_path):
    rows = []
    for index in range(100):
        is_phishing = index % 2 == 0
        host = f"{'verify' if is_phishing else 'news'}{index}.example{index}.com"
        rows.append({
            'URL': f"http://{host}/{'login-account' if is_phishing else 'article'}?id={index}",
            'label': 0 if is_phishing else 1,
        })
    source = tmp_path / 'training.csv'
    model_path = tmp_path / 'model.joblib'
    metrics_path = tmp_path / 'metrics.json'
    pd.DataFrame(rows).to_csv(source, index=False)

    metrics = train(source, model_path, metrics_path)
    assert model_path.is_file() and metrics_path.is_file()
    assert 0 <= metrics['accuracy'] <= 1
    assert sum(sum(row) for row in metrics['confusion_matrix']) == metrics['test_rows']
    bundle = load_bundle(model_path)
    prediction = predict_url('http://verify-now.example999.com/login?account=1', bundle)
    explanation = explain_prediction(bundle, prediction['features'], top_n=3)
    assert len(explanation['features']) == 3
