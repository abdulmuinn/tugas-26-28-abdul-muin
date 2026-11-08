"""Bagian B — definisi model dan evaluasinya (dipanggil oleh pipeline.py).

Library: sklearn (ColumnTransformer, Pipeline, StandardScaler, SimpleImputer, OneHotEncoder,
LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, metrics).
"""
from __future__ import annotations

from data import CATEGORICAL, NUMERIC, SEED  # noqa: F401

# Tiga kandidat yang dibandingkan di MLflow. Data, split, dan preprocessing harus sama untuk ketiganya.
# Format: nama -> (kelas model sklearn, hyperparameter)
CANDIDATES = {
    "logreg": ("LogisticRegression", {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}),
    "random_forest": ("RandomForestClassifier", {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5,
                                                 "class_weight": "balanced", "random_state": SEED}),
    "gradient_boosting": ("GradientBoostingClassifier", {"n_estimators": 150, "max_depth": 2,
                                                         "learning_rate": 0.05, "random_state": SEED}),
}


def build_model(name: str):
    """Membangun preprocessing dan model klasifikasi."""

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
    )

    # Daftar jenis model yang diperbolehkan
    model_classes = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }

    if name not in CANDIDATES:
        raise ValueError(f"Model tidak dikenal: {name}")

    # Ambil konfigurasi dari template mentor
    class_name, params = CANDIDATES[name]

    # Preprocessing fitur numerik
    num_pipeline = StandardScaler()

    # Preprocessing fitur kategori
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore"
        )),
    ])

    # Gabungkan kedua jenis preprocessing
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERIC),
        ("cat", cat_pipeline, CATEGORICAL),
    ])

    # Bangun classifier dengan hyperparameter asli
    classifier = model_classes[class_name](**params)

    # Preprocessing dan model dalam satu pipeline
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", classifier),
    ])

    return model


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Evaluasi model untuk kelas bad (1)."""

    from sklearn.metrics import (
        roc_auc_score,
        recall_score,
        precision_score,
        f1_score,
    )

    # Probabilitas kelas bad
    prob_bad = model.predict_proba(X)[:, 1]

    # Ubah probabilitas menjadi kelas prediksi
    y_pred = (prob_bad >= threshold).astype(int)

    # Hitung seluruh metrik
    metrics = {
        "roc_auc": round(
            float(roc_auc_score(y, prob_bad)), 4
        ),
        "recall_bad": round(
            float(recall_score(y, y_pred, zero_division=0)), 4
        ),
        "precision_bad": round(
            float(precision_score(y, y_pred, zero_division=0)), 4
        ),
        "f1_bad": round(
            float(f1_score(y, y_pred, zero_division=0)), 4
        ),
    }

    return metrics