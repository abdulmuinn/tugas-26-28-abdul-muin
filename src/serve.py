"""Serving model MLflow champion dengan pencatatan setiap request.

Alur: validasi -> inferensi -> keputusan -> catat status / latency / input.
"""
from __future__ import annotations

import csv
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from data import FEATURES, ROOT, validate_record

LOG_PATH = ROOT / "logs" / "predictions.csv"
LOG_COLUMNS = ["timestamp", "request_id", "batch", "status", "error", "latency_ms", "prob_bad", "decision"] + FEATURES
MODEL_URI = "models:/credit-risk@champion"


def decide(prob_bad: float) -> str:
    """K05 Pasal 2: di bawah 0,3 setuju; 0,3 s.d. 0,6 review; di atas 0,6 tolak."""
    if not 0 <= float(prob_bad) <= 1:
        raise ValueError("prob_bad harus berada di rentang 0 sampai 1")
    if prob_bad < 0.3:
        return "setuju"
    if prob_bad <= 0.6:
        return "review"
    return "tolak"


class CreditScorer:
    def __init__(self, model_uri: str = MODEL_URI):
        """Set MLflow lokal; muat champion sekali saja dan siapkan header log."""
        import mlflow
        import mlflow.sklearn

        tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_uri = model_uri
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
            with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=LOG_COLUMNS).writeheader()

    def predict(self, record: dict, batch: str = "live", request_id: str | None = None) -> dict:
        """Catat satu request walau input validasinya gagal; output tanpa kolom fitur."""
        start = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "batch": batch,
            "status": "ok",
            "error": "",
            "latency_ms": None,
            "prob_bad": None,
            "decision": "",
        }
        try:
            validate_record(record)
            data = pd.DataFrame([{field: record[field] for field in FEATURES}], columns=FEATURES)
            prob = float(self.model.predict_proba(data)[0, 1])
            row["prob_bad"] = round(prob, 4)
            row["decision"] = decide(row["prob_bad"])
        except ValueError as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            row["latency_ms"] = round((time.perf_counter() - start) * 1000, 4)
            full_row = {**row, **{field: record.get(field) for field in FEATURES}}
            with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=LOG_COLUMNS).writerow(full_row)
        return row
