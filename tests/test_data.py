import numpy as np
import pandas as pd
import pytest
import pyreadstat

from medpredict_ml.data import prepare_frame, read_table, split_participants, validate_table


def test_xpt_zero_weights_stay_zero_and_cannot_enter_cohort(tmp_path):
    path = tmp_path / "synthetic.xpt"
    fixture = tables()["P_GLU"]
    pyreadstat.write_xport(fixture, path, file_format_version=5)
    restored = read_table(path, "P_GLU")
    np.testing.assert_array_equal(restored.WTSAFPRP, fixture.WTSAFPRP)
    assert (restored.WTSAFPRP == 0).sum() == 1
    fixture.loc[0, "WTSAFPRP"] = 2.**-260
    pyreadstat.write_xport(fixture, path, file_format_version=5)
    with pytest.raises(ValueError, match="zero decoding"):
        read_table(path, "P_GLU")


def tables():
    ids = [1, 2, 3, 4]
    return {
        "P_DEMO": pd.DataFrame({"SEQN": ids, "RIDAGEYR": [52, 80, 40, 30],
            "RIAGENDR": [1, 2, 2, 1], "SDMVSTRA": [1]*4, "SDMVPSU": [1, 2, 1, 2]}),
        "P_BMX": pd.DataFrame({"SEQN": ids, "BMXBMI": [27.4]*4}),
        "P_TCHOL": pd.DataFrame({"SEQN": ids, "LBDTCSI": [5.6]*4}),
        "P_GLU": pd.DataFrame({"SEQN": ids, "LBDGLUSI": [5.9]*4, "WTSAFPRP": [2., 3., 0., 1.]}),
        "P_BPXO": pd.DataFrame({"SEQN": ids, "BPXOSY1": [120., 140., 110., 125.],
            "BPXOSY2": [124., 142., 112., np.nan], "BPXOSY3": [np.nan, 144., 114., np.nan],
            "BPXODI1": [80.]*4, "BPXODI2": [80.]*4, "BPXODI3": [np.nan, 80., 80., np.nan]}),
        "P_BPQ": pd.DataFrame({"SEQN": ids, "BPQ050A": [1., 2., np.nan, 2.]}),
    }


CONFIG = {"age_min": 18, "age_max": 79, "minimum_sbp_readings": 2, "seed": 2026}


def test_duplicate_ids_are_rejected_before_join():
    with pytest.raises(ValueError, match="duplicate"):
        validate_table(pd.DataFrame({"SEQN": [1., 1.], "x": [2., 3.]}), ["SEQN", "x"], "fixture")


def test_missing_columns_and_non_numeric_values_fail():
    with pytest.raises(ValueError, match="columns"):
        validate_table(pd.DataFrame({"SEQN": [1]}), ["SEQN", "x"], "fixture")
    with pytest.raises(ValueError, match="numeric"):
        validate_table(pd.DataFrame({"SEQN": [1], "x": ["bad"]}), ["SEQN", "x"], "fixture")


def test_cohort_fasting_eligibility_and_mean_target():
    frame, audit = prepare_frame(tables(), CONFIG)
    assert frame.SEQN.tolist() == [1]
    assert frame.target.tolist() == [122.]
    assert frame.sex.tolist() == [0.]
    assert audit["attrition"][0]["remaining"] == 4
    assert audit["attrition"][-1]["remaining"] == 1
    assert audit["missing_after_merge"]["glucose"] == 0


def test_sbp_reading_below_diastolic_cannot_enter_target():
    data = tables()
    data["P_BPXO"].loc[0, "BPXODI1"] = 130.
    frame, audit = prepare_frame(data, CONFIG)
    assert frame.empty
    assert audit["invalid_sbp_readings"] == 1


def test_missing_target_is_not_imputed_and_nonpositive_input_is_excluded():
    data = tables()
    data["P_GLU"].loc[0, "LBDGLUSI"] = -1.
    frame, _ = prepare_frame(data, CONFIG)
    assert frame.empty


def test_split_is_disjoint_deterministic_and_preserves_every_participant():
    ids = np.arange(1000)
    first = split_participants(ids, 2026)
    second = split_participants(ids, 2026)
    assert np.array_equal(first, second)
    assert pd.Series(first).value_counts().to_dict() == {"train": 700, "validation": 150, "test": 150}
    for role in ["train", "validation", "test"]:
        assert set(ids[first == role]).isdisjoint(set(ids[first != role]))
    with pytest.raises(ValueError, match="unique"):
        split_participants(np.array([1, 1, 2]), 2026)


def test_split_respects_registered_fractions():
    roles = split_participants(np.arange(100), 2026, .2, .1)
    assert pd.Series(roles).value_counts().to_dict() == {"train": 70, "validation": 20, "test": 10}
    with pytest.raises(ValueError, match="fractions"):
        split_participants(np.arange(100), 2026, .6, .6)

