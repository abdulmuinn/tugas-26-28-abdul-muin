"""Simulasi 300 request minggu_1 dan 300 request minggu_2 dengan enam input rusak."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, SEED, load_data, split  # noqa: E402


def make_week2(X):
    """Buat salinan yang bergeser: age -10, amount x1,8, duration +12, purpose 35%."""
    import numpy as np

    changed = X.copy(deep=True)
    changed["age"] = np.maximum(19, changed["age"] - 10)
    changed["credit_amount"] = np.rint(changed["credit_amount"] * 1.8).astype(int)
    changed["duration"] = np.minimum(72, changed["duration"] + 12)
    rng = np.random.default_rng(SEED)
    selected = rng.choice(len(changed), size=round(len(changed) * 0.35), replace=False)
    changed.iloc[selected, changed.columns.get_loc("purpose")] = "business"
    return changed


def main() -> None:
    """Simulasi, reset log, simpan label secara terpisah dan tampilkan rekap status."""
    import pandas as pd
    from serve import CreditScorer, LOG_PATH

    _, X_test, _, y_test = split(load_data())
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)
    scorer = CreditScorer()
    labels = []
    status_rows = []

    for i, (record, truth) in enumerate(zip(X_test.to_dict("records"), y_test.to_numpy())):
        rid = f"1-{i:04d}"
        status_rows.append(scorer.predict(record, batch="minggu_1", request_id=rid))
        labels.append({"request_id": rid, "bad": int(truth)})

    week2 = make_week2(X_test).to_dict("records")
    for i, (record, truth) in enumerate(zip(week2, y_test.to_numpy())):
        if i in (10, 110):
            record["age"] = None
        elif i in (20, 120):
            record["credit_amount"] = -500
        elif i == 30:
            record["duration"] = "dua belas"
        elif i == 40:
            record["purpose"] = ""
        rid = f"2-{i:04d}"
        status_rows.append(scorer.predict(record, batch="minggu_2", request_id=rid))
        labels.append({"request_id": rid, "bad": int(truth)})

    pd.DataFrame(labels, columns=["request_id", "bad"]).to_csv(
        ROOT / "logs" / "labels.csv", index=False
    )
    count = pd.DataFrame(status_rows).groupby(["batch", "status"]).size()
    print("SIMULASI SELESAI — 600 request dikirim")
    print(count.to_string())
    print("Log:", LOG_PATH)
    print("Label:", ROOT / "logs" / "labels.csv")


if __name__ == "__main__":
    main()
