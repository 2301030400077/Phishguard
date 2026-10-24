# Dataset instructions

Download the **PhiUSIIL Phishing URL (Website)** CSV from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset) and save it here as `PhiUSIIL_Phishing_URL_Dataset.csv`.

The UCI data card reports 235,795 URL records, including 134,850 legitimate and 100,945 phishing URLs. Its `label` target uses `1 = legitimate`, `0 = phishing`; `src/train_model.py` maps to `0 = legitimate`, `1 = phishing`. Only `URL` and `label` are used. Page-derived and precomputed dataset features are intentionally excluded so that training and live prediction share the same URL parser.

Run training from the repository root with `python -m src.train_model`. The data is not included in this repository. The dataset is CC BY 4.0; cite Prasad & Chandra (2024), DOI `10.1016/j.cose.2023.103545`.
