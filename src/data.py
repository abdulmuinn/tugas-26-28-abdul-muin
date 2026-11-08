"""Data German Credit (credit risk) — dipakai Bagian B dan C.

Sumber: https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk
Simpan file aslinya sebagai data/credit/german_credit_data.csv
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit" / "german_credit_data.csv"
SEED = 42

NUMERIC = ["age", "credit_amount", "duration"]
CATEGORICAL = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "bad"  # 1 = risiko buruk (gagal bayar), 0 = baik



def load_data(path: Path = DATA_PATH):
    """Membaca dan menyiapkan dataset German Credit."""
    import pandas as pd

    # 1. Periksa keberadaan file
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. "
            "Unduh German Credit Data with Risk dari Kaggle "
            "dan simpan di data/credit/german_credit_data.csv"
        )

    # 2. Baca CSV, kolom pertama adalah indeks
    df = pd.read_csv(path, index_col=0)

    # 3. Normalisasi nama kolom
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # 4. Job merupakan kategori, bukan angka kontinu
    df["job"] = df["job"].astype(str)

    # 5. Tangani missing values sesuai aturan tugas
    for col in ["saving_accounts", "checking_account"]:
        df[col] = df[col].fillna("unknown")

    # 6. Buat label biner
    df[TARGET] = (
        df["risk"].str.lower().eq("bad").astype(int)
    )

    # 7. Hanya kembalikan fitur dan target yang dibutuhkan
    return df[FEATURES + [TARGET]].copy()




def split(df):
    """Membagi dataset 70/30 secara stratified."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    return X_train, X_test, y_train, y_test




def data_version(path: Path = DATA_PATH) -> str:
    """Menghasilkan fingerprint SHA-256 dataset."""
    import hashlib

    file_bytes = path.read_bytes()

    hash_value = hashlib.sha256(file_bytes).hexdigest()

    return hash_value[:12]




def validate_record(record: dict) -> None:
    """Validasi input sebelum model memprediksi."""
    from numbers import Real
    from math import isfinite

    # 1. Periksa field yang hilang atau kosong
    empty_fields = [
        field
        for field in FEATURES
        if field not in record
        or record[field] is None
        or (
            isinstance(record[field], str)
            and not record[field].strip()
        )
    ]

    if empty_fields:
        raise ValueError(f"field kosong: {empty_fields}")

    # 2. Periksa fitur numerik
    for field in NUMERIC:
        value = record[field]

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{field} harus angka positif, dapat {value}"
            )

    return None

