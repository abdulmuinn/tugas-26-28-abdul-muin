import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import answer, load_chunks, rrf  # noqa: E402


def test_chunking_per_pasal():
    chunks = load_chunks()
    assert len(chunks) == 24                      # 8 dokumen x 3 pasal
    assert chunks[0]["chunk_id"] == "K01-P1"


def test_rrf_contoh_modul():
    # contoh di dokumen tugas: A(1,5), B(2,1), C(3,2) -> urutan B, C, A
    dense = ["A", "B", "C"]
    bm25 = ["B", "C", "X", "Y", "A"]
    assert rrf([dense, bm25])[:3] == ["B", "C", "A"]


def test_pertanyaan_kosong_fallback():
    for q in ["", "     "]:
        r = answer(q)
        assert r["fallback"] is True and r["sources"] == []
