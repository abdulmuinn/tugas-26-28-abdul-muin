"""Monitoring input drift, operasi serving, performa, alert, dashboard, dan SOP trigger."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CATEGORICAL, NUMERIC, ROOT, load_data, split  # noqa: E402

PSI_LIMIT, KS_P_LIMIT, JS_LIMIT = 0.2, 0.05, 0.1
ERROR_RATE_LIMIT, P95_LATENCY_MS_LIMIT, AUC_LIMIT = 0.01, 200, 0.70
EPS = 1e-4


def psi(reference, current, bins: int = 10) -> float:
    """PSI dengan kuantil referensi dan perlindungan bin berprobabilitas nol."""
    import numpy as np

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0 or bins < 1:
        raise ValueError("PSI membutuhkan dua sampel tidak kosong dan bins >= 1")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cutoffs = edges[1:-1]
    boundaries = np.r_[-np.inf, cutoffs, np.inf]
    p = np.histogram(ref, bins=boundaries)[0] / len(ref) + EPS
    q = np.histogram(cur, bins=boundaries)[0] / len(cur) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def js_distance(reference, current) -> float:
    """JS distance antar kategori dengan alfabet kategori gabungan dan base=2."""
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import jensenshannon

    ref = pd.Series(reference, dtype="string").fillna("unknown")
    cur = pd.Series(current, dtype="string").fillna("unknown")
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("JS membutuhkan dua sampel tidak kosong")
    categories = sorted(set(ref.tolist()) | set(cur.tolist()))
    p = ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    q = cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    return float(jensenshannon(p, q, base=2))


def drift_table(reference, current, batch: str):
    """Metrik PSI+KS untuk numerik dan JS untuk kategori, satu baris per fitur."""
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp

    rows = []
    for field in NUMERIC:
        ref = pd.to_numeric(reference[field], errors="coerce").dropna()
        cur = pd.to_numeric(current[field], errors="coerce").dropna()
        if ref.empty or cur.empty:
            raise ValueError(f"Tidak ada data numerik untuk {field} ({batch})")
        ps = psi(ref, cur)
        ks = ks_2samp(ref, cur)
        rows.append({
            "batch": batch, "feature": field, "type": "numerik",
            "psi": ps, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "js": np.nan, "drift": bool(ps > PSI_LIMIT or ks.pvalue < KS_P_LIMIT),
        })
    for field in CATEGORICAL:
        dist = js_distance(reference[field], current[field])
        rows.append({
            "batch": batch, "feature": field, "type": "kategori",
            "psi": np.nan, "ks_stat": np.nan, "ks_p": np.nan,
            "js": dist, "drift": bool(dist > JS_LIMIT),
        })
    return pd.DataFrame(rows)


def ops_metrics(log, labels):
    """Satu baris metrik per batch; AUC hanya dihitung jika label tersedia dua kelas."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    rows = []
    for batch, group in log.groupby("batch", sort=True):
        requests = len(group)
        valid = group.loc[group["status"] == "ok"].copy()
        latencies = pd.to_numeric(group["latency_ms"], errors="coerce").dropna()
        times = pd.to_datetime(group["timestamp"], errors="coerce", utc=True).dropna()
        elapsed = (times.max() - times.min()).total_seconds() if len(times) >= 2 else 0.0
        throughput = requests / elapsed if elapsed > 0 else 0.0
        valid["prob_bad"] = pd.to_numeric(valid["prob_bad"], errors="coerce")
        labeled = valid[["request_id", "prob_bad"]].merge(
            labels[["request_id", "bad"]], how="inner", on="request_id", validate="one_to_one"
        ).dropna(subset=["prob_bad", "bad"])
        auc = (
            float(roc_auc_score(labeled["bad"].astype(int), labeled["prob_bad"]))
            if labeled["bad"].nunique() >= 2 else np.nan
        )
        rows.append({
            "batch": batch,
            "requests": requests,
            "error_rate": float((group["status"] == "error").mean()),
            "p50_latency_ms": float(latencies.median()) if len(latencies) else np.nan,
            "p95_latency_ms": float(latencies.quantile(0.95)) if len(latencies) else np.nan,
            "throughput_rps": float(throughput),
            "mean_prob_bad": float(valid["prob_bad"].mean()) if not valid.empty else np.nan,
            "share_tolak": float((valid["decision"] == "tolak").mean()) if not valid.empty else np.nan,
            "auc": auc,
        })
    return pd.DataFrame(rows)


def alerts(ops, drift) -> list[str]:
    """Bunyikan alert untuk batas error, p95, AUC, atau drift."""
    import pandas as pd

    messages = []
    for row in ops.itertuples(index=False):
        if row.error_rate > ERROR_RATE_LIMIT:
            messages.append(f"{row.batch}: error rate {row.error_rate:.1%} > {ERROR_RATE_LIMIT:.0%}")
        if pd.notna(row.p95_latency_ms) and row.p95_latency_ms > P95_LATENCY_MS_LIMIT:
            messages.append(f"{row.batch}: p95 latency {row.p95_latency_ms:.1f} ms > {P95_LATENCY_MS_LIMIT} ms")
        if pd.notna(row.auc) and row.auc < AUC_LIMIT:
            messages.append(f"{row.batch}: AUC {row.auc:.4f} < {AUC_LIMIT:.2f}")
    for batch, table in drift.groupby("batch", sort=True):
        changed = table.loc[table["drift"].astype(bool), "feature"].tolist()
        if changed:
            messages.append(f"{batch}: drift pada {', '.join(changed)}")
    return messages


def dashboard(reference, log, ops, drift, path: Path) -> None:
    """Dashboard PNG 2x3: age, amount, PSI, JS, error rate, keputusan kredit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    good = log.loc[log["status"] == "ok"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    batches = sorted(log["batch"].dropna().unique())
    for ax, field in zip(axes[0, :2], ["age", "credit_amount"]):
        ref = pd.to_numeric(reference[field], errors="coerce").dropna()
        ax.hist(ref, bins=20, alpha=0.42, density=True, label="training reference")
        for batch in batches:
            cur = pd.to_numeric(good.loc[good["batch"] == batch, field], errors="coerce").dropna()
            ax.hist(cur, bins=20, alpha=0.35, density=True, label=batch)
        ax.set_title(f"Distribusi {field}")
        ax.set_xlabel(field)
        ax.set_ylabel("Densitas")
        ax.legend(fontsize=8)

    for ax, metric, category, limit, title in [
        (axes[0, 2], "psi", "numerik", PSI_LIMIT, "PSI fitur numerik"),
        (axes[1, 0], "js", "kategori", JS_LIMIT, "JS distance fitur kategori"),
    ]:
        subset = drift.loc[drift["type"] == category]
        features = NUMERIC if category == "numerik" else CATEGORICAL
        x = np.arange(len(features))
        width = 0.8 / max(1, len(batches))
        for i, batch in enumerate(batches):
            vals = subset.loc[subset["batch"] == batch].set_index("feature")[metric]
            ax.bar(x + (i - (len(batches) - 1) / 2) * width,
                   [float(vals.get(f, np.nan)) for f in features], width, label=batch)
        ax.axhline(limit, linestyle="--", linewidth=1.4, label=f"batas {limit}")
        ax.set_xticks(x, features, rotation=30, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(ops["batch"], ops["error_rate"] * 100)
    ax.axhline(ERROR_RATE_LIMIT * 100, linestyle="--", label="batas 1%")
    ax.set_title("Error rate per batch")
    ax.set_ylabel("Persen (%)")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    table = pd.crosstab(good["batch"], good["decision"], normalize="index") * 100
    table.reindex(index=batches, fill_value=0).reindex(columns=["setuju", "review", "tolak"], fill_value=0).plot(
        kind="bar", stacked=True, ax=ax
    )
    ax.set_title("Porsi keputusan kredit")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Persen (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)

    fig.suptitle("Bank Nusantara — Model Monitoring", fontsize=17)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Proses seluruh log, bandingkan distribusi, dan buat empat deliverable laporan."""
    import json
    import pandas as pd

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs"
    prediction_path, label_path = logs / "predictions.csv", logs / "labels.csv"
    if not prediction_path.exists() or not label_path.exists():
        raise FileNotFoundError("Jalankan python src/simulate.py terlebih dahulu untuk membuat log dan label.")

    X_train, _, _, _ = split(load_data())
    log = pd.read_csv(prediction_path, dtype={field: "string" for field in CATEGORICAL})
    labels = pd.read_csv(label_path)
    if log.empty:
        raise ValueError("Log prediksi masih kosong")
    good = log.loc[log["status"] == "ok"]
    drift = pd.concat([
        drift_table(X_train, good.loc[good["batch"] == batch], batch)
        for batch in sorted(log["batch"].dropna().unique())
    ], ignore_index=True)
    ops = ops_metrics(log, labels)
    warning_messages = alerts(ops, drift)

    drift.to_csv(reports / "drift_metrics.csv", index=False)
    dashboard(X_train, log, ops, drift, reports / "monitoring_dashboard.png")
    drift_features = drift.loc[drift["drift"] == True, ["batch", "feature"]]
    needs_retrain_review = not drift_features.empty
    retrain_trigger = {
        "retrain": bool(needs_retrain_review),
        "alerts": warning_messages,
        "next_step": (
            "Investigasi drift, validasi kualitas label baru, dan evaluasi kandidat retraining; "
            "jangan otomatis mempromosikan model."
            if needs_retrain_review else "Lanjutkan monitoring; belum ada pemicu drift signifikan."
        ),
    }
    (reports / "retrain_trigger.json").write_text(
        json.dumps(retrain_trigger, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    errors = log.loc[log["status"] == "error", ["request_id", "batch", "error"]].head(10)
    report = [
        "# Monitoring Model Credit Risk — Bank Nusantara",
        "",
        "Referensi: 700 data training; batch simulasi minggu_1 dan minggu_2.",
        "Label simulasi disimpan terpisah; label kredit riil umumnya baru tersedia belakangan.",
        "",
        "## Metrik Operasional dan Kualitas",
        ops.round(4).to_markdown(index=False), "",
        "## Drift Fitur (PSI / KS / JS)",
        drift.round(4).to_markdown(index=False), "",
        "## Alert",
        *([f"- {message}" for message in warning_messages] if warning_messages else ["Tidak ada alert."]),
        "", "## Contoh Input Gagal",
        errors.to_markdown(index=False) if not errors.empty else "Tidak ada input gagal.", "",
        "## Interpretasi dan Tindak Lanjut",
        "Drift menunjukkan perubahan sebaran, bukan otomatis penurunan akurasi. ",
        "Periksa terlebih dahulu integritas input, kesesuaian sampel, besaran drift, "
        "ketersediaan label, lalu bandingkan kandidat retrain dengan champion melalui quality gate.",
        "", "![Monitoring Dashboard](monitoring_dashboard.png)", "",
    ]
    (reports / "monitoring_report.md").write_text("\n".join(report), encoding="utf-8")
    print("METRIK OPERASIONAL")
    print(ops.round(4).to_string(index=False))
    print("FITUR DRIFT")
    print(drift.loc[drift["drift"] == True, ["batch", "feature", "psi", "ks_p", "js"]].round(4).to_string(index=False))
    for message in warning_messages:
        print("ALERT", message)
    print("MONITORING SELESAI — 4 file di reports/ berhasil dibuat")


if __name__ == "__main__":
    main()
