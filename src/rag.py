"""Bagian A (Pertemuan 26) — RAG untuk dokumen kebijakan kredit.

Alur: dokumen -> chunk per pasal -> dense + BM25 -> hybrid (RRF) -> rerank -> jawaban + sitasi / fallback.

Library yang dipakai: re, numpy, rank_bm25 (BM25Okapi), sentence_transformers (SentenceTransformer, CrossEncoder).
"""
from __future__ import annotations

from pathlib import Path

from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "rag" / "docs"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FALLBACK_THRESHOLD = 0.35   # ambang top_score dense; sesuaikan dengan datamu dan jelaskan di README
RRF_K = 60
CANDIDATES = 8              # jumlah kandidat hybrid yang dikirim ke reranker
FALLBACK_TEXT = "Maaf, informasi tersebut tidak ada di dokumen kebijakan kredit."


# ---------------------------------------------------------------- 1. chunking

def load_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Membaca dokumen dan memecahnya berdasarkan pasal."""
    import re

    chunks = []
    pattern = re.compile(r"(?m)^Pasal\s+(\d+)\s*:\s*")

    for path in sorted(docs_dir.glob("K*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].removeprefix("# ").strip()
        body = "\n".join(lines[1:])
        matches = list(pattern.finditer(body))

        for index, match in enumerate(matches):
            pasal = int(match.group(1))
            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(body)
            )

            chunks.append({
                "chunk_id": f"{path.stem}-P{pasal}",
                "doc_id": path.stem,
                "pasal": pasal,
                "title": title,
                "text": body[start:end].strip(),
            })

    return chunks




def tokenize(text: str) -> list[str]:
    """Ubah teks menjadi token untuk pencarian BM25."""
    import re

    text = text.lower()
    tokens = re.findall(r"[^\W_]+", text)

    return tokens



# ---------------------------------------------------------------- index (dibuat sekali)

@lru_cache(maxsize=1)
def _index():
    """Membangun dan menyimpan seluruh komponen RAG."""

    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi

    # 1. Ambil seluruh chunk dokumen
    chunks = load_chunks()

    # 2. Gabungkan judul dan isi tiap chunk
    texts = [
        f"{chunk['title']}. {chunk['text']}"
        for chunk in chunks
    ]

    # 3. Buat model embedding
    embedder = SentenceTransformer(EMBED_MODEL)

    # 4. Ubah teks menjadi vektor
    vectors = embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 5. Buat indeks BM25
    tokenized_docs = [
        tokenize(text) for text in texts
    ]

    bm25 = BM25Okapi(tokenized_docs)

    # 6. Siapkan reranker
    reranker = CrossEncoder(RERANK_MODEL)

    return chunks, texts, embedder, vectors, bm25, reranker



# ---------------------------------------------------------------- 2-3. dense dan BM25

def dense_search(question: str) -> tuple[list[int], list[float]]:
    """Cari chunk berdasarkan cosine similarity."""
    import numpy as np

    chunks, texts, embedder, vectors, bm25, reranker = _index()

    query_vector = embedder.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores = np.asarray(vectors @ query_vector).ravel()

    order = np.argsort(-scores, kind="stable")

    return (
        order.tolist(),
        scores[order].astype(float).tolist()
    )




def bm25_search(question: str) -> list[int]:
    """Cari chunk berdasarkan relevansi kata BM25."""
    import numpy as np

    chunks, texts, embedder, vectors, bm25, reranker = _index()

    tokens = tokenize(question)

    scores = bm25.get_scores(tokens)

    order = np.argsort(-scores, kind="stable")

    return order.tolist()



# ---------------------------------------------------------------- 4. hybrid (RRF)

def rrf(rankings: list[list], k: int = RRF_K) -> list:
    """Gabungkan beberapa ranking menggunakan RRF."""

    scores = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if item not in scores:
                scores[item] = 0.0

            scores[item] += 1 / (k + rank)

    return sorted(
        scores,
        key=lambda item: scores[item],
        reverse=True
    )



# ---------------------------------------------------------------- 5. rerank

def rerank(question: str, candidates: list[int]) -> list[int]:
    """Urutkan kandidat menggunakan Cross-Encoder."""
    import numpy as np

    if not candidates:
        return []

    chunks, texts, embedder, vectors, bm25, reranker = _index()

    # Pasangkan pertanyaan dengan setiap kandidat
    pairs = [
        (question, texts[i])
        for i in candidates
    ]

    # Hitung skor relevansi
    scores = np.asarray(
        reranker.predict(pairs)
    ).reshape(-1)

    # Urutkan skor terbesar ke terkecil
    order = np.argsort(-scores, kind="stable")

    return [candidates[i] for i in order]



# ---------------------------------------------------------------- 6. jawaban + sitasi + fallback

def answer(question: str, mode: str = "hybrid_rerank", top_k: int = 3) -> dict:
    """Jawab pertanyaan dengan sumber atau fallback."""
    import time

    start = time.perf_counter()

    # Pertanyaan kosong tidak boleh memuat model
    if not question.strip():
        return {
            "answer": FALLBACK_TEXT,
            "sources": [],
            "top_score": 0.0,
            "fallback": True,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    valid_modes = {"dense", "bm25", "hybrid", "hybrid_rerank"}

    if mode not in valid_modes:
        raise ValueError(f"Mode tidak dikenal: {mode}")

    if top_k < 1:
        raise ValueError("top_k harus minimal 1")

    # Skor fallback selalu berasal dari dense search
    dense_order, dense_scores = dense_search(question)
    top_score = float(dense_scores[0]) if dense_scores else 0.0

    # Tolak jika kemiripan tidak melewati threshold
    if top_score < FALLBACK_THRESHOLD:
        return {
            "answer": FALLBACK_TEXT,
            "sources": [],
            "top_score": top_score,
            "fallback": True,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    # Tentukan urutan berdasarkan mode
    if mode == "dense":
        order = dense_order
    else:
        bm25_order = bm25_search(question)

        if mode == "bm25":
            order = bm25_order
        else:
            hybrid_order = rrf([dense_order, bm25_order])

            if mode == "hybrid":
                order = hybrid_order
            else:
                order = rerank(
                    question,
                    hybrid_order[:CANDIDATES]
                )

    chunks = _index()[0]

    if not order:
        return {
            "answer": FALLBACK_TEXT,
            "sources": [],
            "top_score": top_score,
            "fallback": True,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    best = chunks[order[0]]

    answer_text = (
        f"{best['text']} "
        f"[Sumber: {best['doc_id']}, Pasal {best['pasal']}]"
    )

    sources = [
        chunks[i]["doc_id"]
        for i in order[:top_k]
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "top_score": top_score,
        "fallback": False,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


if __name__ == "__main__":
    for q in ["Berapa denda kalau telat bayar?", "Siapa presiden pertama Indonesia?", ""]:
        print(q or "<kosong>", "->", answer(q))
