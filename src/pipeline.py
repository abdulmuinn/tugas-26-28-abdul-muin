"""Bagian B (Pertemuan 27) — pipeline commit-to-deploy versi lokal.

Urutan: test -> train 3 kandidat (MLflow) -> quality gate -> pilih terbaik
-> register -> promote "champion".

Jalankan dari root repo:
    python src/pipeline.py --reason "commit baru"

Catatan: MLflow 3 mencatat model sebagai Logged Model (model URI models:/m-...)
sehingga kita menyimpan URI yang benar dalam tag run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT  # noqa: E402

EXPERIMENT = "credit-risk"
MODEL_NAME = "credit-risk"
ALIAS = "champion"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def setup_mlflow() -> None:
    """Set tracking URI, buat eksperimen dengan artifact_location lokal jika belum ada."""
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    (ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT,
            artifact_location=(ROOT / "mlruns").resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow: {TRACKING_URI}")


def step_test() -> None:
    """[1/5] Jalankan test otomatis, hentikan pipeline jika gagal."""
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    print("[1/5] TEST:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Pipeline berhenti: pytest gagal (exit={result.returncode}).")


def step_train(reason: str) -> list[dict]:
    """[2/5] Latih semua kandidat, catat parameter/metrik/model dalam MLflow."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from data import data_version, load_data, split
    from train import CANDIDATES, build_model, evaluate

    X_train, X_test, y_train, y_test = split(load_data())
    version = data_version()
    results: list[dict] = []

    for name, (_, hyperparams) in CANDIDATES.items():
        print(f"[2/5] TRAIN: {name}", flush=True)
        with mlflow.start_run(run_name=name) as run:
            model = build_model(name)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test, threshold=0.5)

            params = {
                "model": name,
                "data_version": version,
                "n_train": len(X_train),
                **{f"clf__{key}": value for key, value in hyperparams.items()},
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("reason", reason)

            example = X_train.head(3)
            signature = infer_signature(example, model.predict(example))
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=example,
                signature=signature,
                skops_trusted_types=TRUSTED_TYPES,
            )
            # Di MLflow 3, model hasil log punya URI models:/m-..., bukan
            # selalu dapat ditemukan melalui runs:/<run_id>/model.
            mlflow.set_tag("logged_model_uri", info.model_uri)

            result = {
                "run_id": run.info.run_id,
                "model": name,
                "data_version": version,
                **metrics,
            }
            results.append(result)
            print("  METRICS:", metrics, flush=True)
    return results


def step_gate(results: list[dict]) -> list[dict]:
    """[3/5] Tandai lulus/gagal, simpan seluruh kandidat, hentikan jika tidak ada yang lulus."""
    import mlflow
    from quality_gate import check

    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    client = mlflow.tracking.MlflowClient()
    passed: list[dict] = []
    all_results: list[dict] = []

    for result in results:
        ok, reasons = check(result)
        status = "lulus" if ok else "gagal"
        client.set_tag(result["run_id"], "quality_gate", status)
        all_results.append({**result, "quality_gate": status, "reasons": reasons})
        if ok:
            passed.append(result)
        print(f"[3/5] GATE: {result['model']} -> {status}: {reasons}")

    output = ROOT / "reports" / "candidates.json"
    output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit("Pipeline berhenti: tidak ada kandidat yang lulus quality gate.")
    return passed


def step_select(passed: list[dict]) -> dict:
    """[4/5] Pilih model lulus dengan AUC tertinggi dan simpan metriknya."""
    if not passed:
        raise SystemExit("Tidak ada model yang dapat dipilih.")
    best = max(passed, key=lambda row: row["roc_auc"])
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "best_metrics.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print(f"[4/5] SELECT: {best['model']} (ROC-AUC={best['roc_auc']:.4f})")
    return best


def step_register_promote(best: dict) -> None:
    """[5/5] Registry model terpilih; promote bila champion lama belum ada / AUC meningkat."""
    import mlflow
    from mlflow.exceptions import MlflowException

    client = mlflow.tracking.MlflowClient()
    try:
        previous = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException as exc:
        if getattr(exc, "error_code", "") != "RESOURCE_DOES_NOT_EXIST":
            raise
        previous = None

    old_auc = None
    if previous is not None:
        tags = previous.tags or {}
        if "roc_auc" in tags:
            old_auc = float(tags["roc_auc"])
        elif previous.run_id:
            old_auc = client.get_run(previous.run_id).data.metrics.get("roc_auc")
        if old_auc is None:
            raise RuntimeError("Champion lama tidak punya metrik ROC-AUC; promosi dibatalkan demi keamanan.")

    run = client.get_run(best["run_id"])
    logged_uri = run.data.tags.get("logged_model_uri")
    if not logged_uri:
        raise RuntimeError("URI model tidak ditemukan dalam tag run; registrasi dibatalkan.")

    mv = mlflow.register_model(
        model_uri=logged_uri,
        name=MODEL_NAME,
        tags={
            "roc_auc": str(best["roc_auc"]),
            "source_run_id": best["run_id"],
            "data_version": best["data_version"],
        },
    )
    print(f"[5/5] REGISTER: {MODEL_NAME} versi {mv.version}")

    if previous is None or float(best["roc_auc"]) > float(old_auc):
        client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
        print(f"PROMOTE: {MODEL_NAME}@{ALIAS} -> versi {mv.version}")
    else:
        print(
            "TIDAK PROMOTE: AUC baru "
            f"{best['roc_auc']:.4f} <= champion lama {old_auc:.4f} "
            f"(versi {previous.version})"
        )


def main() -> None:
    """Parse argumen dan jalankan test -> train -> gate -> select -> register/promote."""
    parser = argparse.ArgumentParser(description="Local credit risk MLOps pipeline")
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--skip-tests", action="store_true", help="Hanya untuk debugging")
    args = parser.parse_args()

    setup_mlflow()
    if args.skip_tests:
        print("PERINGATAN: test dilewati. Jangan gunakan hasil ini untuk final submission.")
    else:
        step_test()
    all_results = step_train(args.reason)
    passed = step_gate(all_results)
    best = step_select(passed)
    step_register_promote(best)
    print("PIPELINE SELESAI")


if __name__ == "__main__":
    main()
