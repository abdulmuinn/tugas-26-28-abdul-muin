
"""Evaluasi empat mode RAG Bank Nusantara."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import ROOT, answer

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def hit_at_k(expected: str, sources: list[str], k: int = 3) -> bool:
    """Cek apakah dokumen benar ada pada k sumber teratas."""
    if expected == "NONE":
        return False

    return expected in sources[:k]


def summarize(detail: pd.DataFrame) -> dict:
    """Menghitung empat metrik evaluasi."""

    in_topic = detail[detail["expected_doc"] != "NONE"]
    out_topic = detail[detail["expected_doc"] == "NONE"]

    hit3 = (
        float(in_topic["hit3"].mean())
        if not in_topic.empty else 0.0
    )

    citation_correct = (
        (in_topic["top1"] == in_topic["expected_doc"])
        & (~in_topic["fallback"].astype(bool))
    )

    citation_acc = (
        float(citation_correct.mean())
        if not in_topic.empty else 0.0
    )

    fallback_acc = (
        float(out_topic["fallback"].mean())
        if not out_topic.empty else 0.0
    )

    avg_latency = (
        float(detail["latency_ms"].mean())
        if not detail.empty else 0.0
    )

    return {
        "hit_at_3": hit3,
        "citation_acc": citation_acc,
        "fallback_acc": fallback_acc,
        "avg_latency_ms": avg_latency,
    }


def main() -> None:
    """Jalankan dan simpan evaluasi seluruh mode."""

    eval_path = ROOT / "data" / "rag" / "eval.csv"
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(eval_path)

    # Warm-up agar model loading tidak masuk metrik evaluasi
    answer(str(eval_df.iloc[0]["question"]))

    records = []

    for mode in MODES:
        print(f"Evaluasi mode: {mode}", flush=True)

        for row in eval_df.itertuples(index=False):
            question = str(row.question)
            expected = str(row.expected_doc)

            result = answer(
                question,
                mode=mode,
                top_k=3
            )

            sources = result["sources"]
            fallback = bool(result["fallback"])

            top1 = (
                sources[0]
                if sources and not fallback
                else ""
            )

            records.append({
                "mode": mode,
                "question": question,
                "expected_doc": expected,
                "top1": top1,
                "sources": "|".join(sources),
                "hit3": (
                    hit_at_k(expected, sources, 3)
                    and not fallback
                ),
                "top_score": result["top_score"],
                "fallback": fallback,
                "latency_ms": result["latency_ms"],
            })

    detail = pd.DataFrame(records)

    summary_rows = []

    for mode in MODES:
        mode_detail = detail[detail["mode"] == mode]

        summary_rows.append({
            "mode": mode,
            **summarize(mode_detail)
        })

    summary = pd.DataFrame(summary_rows)

    detail.to_csv(
        report_dir / "rag_eval_detail.csv",
        index=False
    )

    summary.to_csv(
        report_dir / "rag_eval_summary.csv",
        index=False
    )

    print("\nHASIL EVALUASI RAG")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()
