import numpy as np
import pandas as pd
import pytest

from medpredict_ml.classification.data import cohort, split, guard_unopened


def tables(n=120):
    ids = np.arange(n)
    return {
        "P_DEMO": pd.DataFrame({"SEQN": ids, "RIDAGEYR": 30, "RIAGENDR": np.where(ids % 2, 2, 1),
            "RIDEXPRG": np.where(ids % 2, 2, np.nan), "SDMVSTRA": 1, "SDMVPSU": 1 + ids % 2}),
        "P_BMX": pd.DataFrame({"SEQN": ids, "BMXBMI": 25.}),
        "P_TCHOL": pd.DataFrame({"SEQN": ids, "LBDTCSI": 5.}),
        "P_GLU": pd.DataFrame({"SEQN": ids, "LBDGLUSI": 5.5, "WTSAFPRP": 100.}),
        "P_GHB": pd.DataFrame({"SEQN": ids, "LBXGH": np.where(ids % 3, 5.5, 6.5)}),
        "P_DIQ": pd.DataFrame({"SEQN": ids, "DIQ010": 2, "DIQ050": np.nan, "DIQ070": np.nan, "DIQ160": 2})}


def test_unknown_pregnancy_is_not_assumed_negative_and_documented_skips_are_explicit():
    source = tables()
    source["P_DEMO"].loc[1, "RIDEXPRG"] = np.nan
    source["P_DIQ"].loc[2, "DIQ160"] = 9
    source["P_DIQ"].loc[3, "DIQ070"] = 1
    source["P_DIQ"].loc[4, "DIQ010"] = 9
    source["P_GHB"].loc[5, "LBXGH"] = np.nan
    frame, audit = cohort(source)
    assert set(range(1, 6)).isdisjoint(frame.SEQN)
    assert 0 in set(frame.SEQN)
    assert frame.loc[frame.SEQN == 0, "target"].item() == 1
    assert "LBXGH" not in frame and "DIQ010" not in frame
    assert "unassessed" in " ".join(audit["limitations"]) or "not establish" in " ".join(audit["limitations"])


def test_duplicate_participant_fails_before_join():
    source = tables()
    source["P_GHB"].loc[1, "SEQN"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        cohort(source)


def test_split_is_repeatable_disjoint_and_contains_both_classes():
    frame, _ = cohort(tables())
    a, b = split(frame, 2026), split(frame, 2026)
    pd.testing.assert_series_equal(a, b)
    assert a.notna().all()
    assert set(a) == {"train", "calibration", "threshold", "test"}
    for name in a.unique():
        assert set(frame.loc[a == name, "target"]) == {0, 1}
    frame.loc[1, "SEQN"] = 0
    with pytest.raises(ValueError, match="Unique"):
        split(frame, 2026)


def test_opened_test_stops_any_new_selection(tmp_path):
    guard_unopened(tmp_path)
    (tmp_path / "test_opened.json").write_text('{}')
    with pytest.raises(RuntimeError, match="already opened"):
        guard_unopened(tmp_path)
