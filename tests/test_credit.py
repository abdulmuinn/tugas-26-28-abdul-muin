import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import DATA_PATH, data_version, load_data, validate_record  # noqa: E402
from monitor import psi  # noqa: E402
from quality_gate import check  # noqa: E402


def test_data_dan_target():
    df = load_data()
    assert len(df) == 1000
    assert set(df["bad"].unique()) == {0, 1}


def test_data_version_stabil():
    assert data_version(DATA_PATH) == data_version(DATA_PATH)
    assert len(data_version(DATA_PATH)) == 12


def test_quality_gate_bisa_gagal():
    assert check({"roc_auc": 0.80, "recall_bad": 0.60})[0] is True
    assert check({"roc_auc": 0.60, "recall_bad": 0.60})[0] is False


def test_validate_record_menolak_input_rusak():
    good = load_data().drop(columns="bad").iloc[0].to_dict()
    validate_record(good)
    with pytest.raises(ValueError):
        validate_record({**good, "age": None})


def test_psi_data_sama_mendekati_nol():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 0.01
