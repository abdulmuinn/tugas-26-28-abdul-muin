"""Bagian B — quality gate: model hanya boleh dipromosikan kalau lulus batas minimal.

Jalankan sendiri: python src/quality_gate.py   (membaca reports/best_metrics.json)
Exit code 0 = lulus, 1 = gagal.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = {"roc_auc": 0.70, "recall_bad": 0.50}



def check(metrics: dict, gates: dict = GATES) -> tuple[bool, list[str]]:
    """Memeriksa apakah model memenuhi seluruh quality gate."""
    import math

    reasons = []

    for metric_name, minimum in gates.items():
        value = metrics.get(metric_name)

        # Metrik harus tersedia dan valid
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            reasons.append(f"{metric_name} tidak valid")
            continue

        # Metrik gagal jika lebih kecil dari batas
        if value < minimum:
            reasons.append(
                f"{metric_name} {value:.3f} < {minimum}"
            )

    passed = len(reasons) == 0

    return passed, reasons




def main() -> int:
    """Membaca metrik champion kandidat dan memeriksa gate."""
    import json

    metrics_path = ROOT / "reports" / "best_metrics.json"

    if not metrics_path.exists():
        print(
            "QUALITY GATE: GAGAL -> "
            "best_metrics.json belum tersedia"
        )
        return 1

    try:
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
        passed, reasons = check(metrics)

    except (OSError, ValueError, TypeError) as exc:
        print(f"QUALITY GATE: GAGAL -> {exc}")
        return 1

    if passed:
        print("QUALITY GATE: LULUS")
        return 0

    print("QUALITY GATE: GAGAL -> " + "; ".join(reasons))
    return 1



if __name__ == "__main__":
    import sys
    sys.exit(main())
