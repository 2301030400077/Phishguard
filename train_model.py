"""Train and evaluate the URL-only Random Forest pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.feature_extraction import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_MODEL = ROOT / "models" / "phishguard_model.joblib"
DEFAULT_METRICS = ROOT / "models" / "metrics.json"

# Curated clean-domain examples help the text model distinguish well-known
# short domains from phishing lookalikes (for example, github.com vs 2fgithub.com).
LEGITIMATE_SEED_HOSTS = (
    "github.com", "google.com", "wikipedia.org", "microsoft.com", "apple.com",
    "amazon.com", "cloudflare.com", "openai.com", "mozilla.org", "python.org",
    "stackoverflow.com", "reddit.com", "linkedin.com", "nytimes.com", "bbc.com",
    "w3.org", "ietf.org", "mit.edu", "stanford.edu", "ubuntu.com",
)


def _label_to_phishing(value: object) -> int | None:
    """Normalize PhiUSIIL labels (0=phishing, 1=legitimate) and common text labels."""
    text = str(value).strip().lower()
    if text in {"0", "0.0", "phishing", "phish", "malicious", "bad"}:
        return 1
    if text in {"1", "1.0", "legitimate", "legit", "benign", "safe", "good"}:
        return 0
    return None


def _valid_dataset_url(value: str) -> bool:
    if any(char.isspace() for char in value):
        return False
    candidate = value if "://" in value else "//" + value
    try:
        parts = urlsplit(candidate)
        return parts.scheme.lower() in {"", "http", "https"} and bool(parts.hostname) and (parts.port is None or 0 < parts.port < 65536)
    except ValueError:
        return False


def _prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {col.strip().lower(): col for col in frame.columns}
    if "url" not in columns or "label" not in columns:
        raise ValueError("Dataset CSV must contain URL and label columns.")
    working = frame[[columns["url"], columns["label"]]].rename(
        columns={columns["url"]: "url", columns["label"]: "label"}
    )
    input_rows = len(working)
    working["url"] = working["url"].astype("string").str.strip()
    working = working.dropna(subset=["url", "label"])
    working = working[working["url"].str.len().between(4, 2048)]
    working = working[working["url"].map(_valid_dataset_url)]
    working["target"] = working["label"].map(_label_to_phishing)
    working = working.dropna(subset=["target"])
    working["target"] = working["target"].astype(int)
    # Remove all URLs with conflicting labels, then keep one of each repeated URL.
    conflicts = working.groupby("url")["target"].nunique()
    conflict_urls = set(conflicts[conflicts > 1].index)
    working = working[~working["url"].isin(conflict_urls)].drop_duplicates("url", keep="first")
    return working


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    frame = pd.read_csv(csv_path, usecols=lambda col: col.strip().lower() in {"url", "label"})
    input_rows = len(frame)
    working = _prepare_rows(frame)
    if working["target"].nunique() != 2 or len(working) < 10:
        raise ValueError("After cleaning, at least 10 rows from both classes are required.")
    features = pd.DataFrame(
        [extract_features(url) for url in working["url"]], columns=FEATURE_NAMES, index=working.index
    )
    labels = working["target"].astype(int)
    metadata = {
        "source_file": Path(csv_path).name,
        "input_rows": int(input_rows),
        "clean_rows": int(len(working)),
        "removed_rows": int(input_rows - len(working)),
        "class_counts_phishing_1": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
    }
    return features, labels, metadata


def _host_group(url: str) -> str:
    candidate = url if "://" in url else "//" + url
    try:
        return (urlsplit(candidate).hostname or url).lower().rstrip(".")
    except ValueError:
        return url


def train(csv_path: str | Path = DEFAULT_DATA, model_path: str | Path = DEFAULT_MODEL,
          metrics_path: str | Path = DEFAULT_METRICS) -> dict:
    features, labels, metadata = load_dataset(csv_path)
    # Re-align the cleaned URL list to the feature frame's indexes after duplicate removal.
    frame = pd.read_csv(csv_path, usecols=lambda col: col.strip().lower() in {"url", "label"})
    frame = _prepare_rows(frame)
    urls = frame.loc[features.index, "url"].astype(str)
    groups = urls.map(_host_group).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = next(splitter.split(features, labels, groups))
    x_train, x_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Grouped split did not preserve both classes. Review host grouping or dataset.")

    explanation_model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=42, n_jobs=1,
    )
    explanation_model.fit(x_train, y_train)

    # A URL character model can learn host spelling and brand impersonation
    # patterns that the original aggregate lexical features cannot represent.
    seed_urls = []
    seen_hosts = set(groups)
    for host in LEGITIMATE_SEED_HOSTS:
        if not ({host, f"www.{host}"} & seen_hosts):
            seed_urls.extend((f"https://{host}/", f"https://{host}/about",
                              f"https://{host}/contact", f"https://www.{host}/"))
    train_urls = pd.concat([urls.iloc[train_index].reset_index(drop=True), pd.Series(seed_urls)], ignore_index=True)
    train_labels = np.concatenate([y_train.to_numpy(), np.zeros(len(seed_urls), dtype=int)])
    sample_weights = np.concatenate([np.ones(len(y_train)), np.full(len(seed_urls), 100.0)])
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2, max_features=200_000,
        sublinear_tf=True, dtype=np.float32,
    )
    x_text_train = vectorizer.fit_transform(train_urls)
    model = LogisticRegression(
        C=4, class_weight="balanced", solver="saga", max_iter=100,
        tol=0.002, random_state=42, n_jobs=1,
    )
    model.fit(x_text_train, train_labels, sample_weight=sample_weights)
    x_text_test = vectorizer.transform(urls.iloc[test_index])
    predicted = model.predict(x_text_test)
    probabilities = model.predict_proba(x_text_test)[:, list(model.classes_).index(1)]
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["legitimate (0)", "phishing (1)"],
        "positive_class": "phishing (1)",
        "split": "StratifiedGroupKFold, 5 folds; first fold held out, grouped by hostname",
        "classifier": "character 2-5 gram TF-IDF + class-weighted logistic regression",
        "curated_legitimate_seed_urls": int(len(seed_urls)),
        "random_state": 42,
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "dataset": metadata,
    }
    background = x_train.sample(n=min(100, len(x_train)), random_state=42).reset_index(drop=True)
    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "explanation_model": explanation_model,
        "feature_names": FEATURE_NAMES,
        "shap_background": background,
        "metrics": metrics,
    }
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with URL and label columns")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    args = parser.parse_args()
    train(args.data, args.model, args.metrics)


if __name__ == "__main__":
    main()
