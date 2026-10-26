from __future__ import annotations

import datetime as dt
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from .common import FEATURES, ROOT, config, guard_test_unopened, sha256, write_json

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
COLUMNS = {
    "P_DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "SDMVSTRA", "SDMVPSU"],
    "P_BMX": ["SEQN", "BMXBMI"],
    "P_TCHOL": ["SEQN", "LBDTCSI"],
    "P_GLU": ["SEQN", "LBDGLUSI", "WTSAFPRP"],
    "P_BPXO": ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"],
    "P_BPQ": ["SEQN", "BPQ050A"],
}
RENAME = {"RIDAGEYR": "age", "RIAGENDR": "sex", "BMXBMI": "bmi", "LBDTCSI": "totalCholesterol",
          "LBDGLUSI": "glucose", "WTSAFPRP": "weight", "SDMVSTRA": "stratum", "SDMVPSU": "psu", "BPQ050A": "medication"}


def valid_transport(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(80).startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!")


def download() -> None:
    destination = ROOT / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in COLUMNS:
        path = destination / f"{name}.xpt"
        url = BASE + path.name
        if not path.exists() or not valid_transport(path):
            temporary = path.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "MedPredict-University-Research/0.1"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    size = 0
                    while chunk := response.read(128 * 1024):
                        size += len(chunk)
                        if size > 30_000_000:
                            raise ValueError("Unexpectedly large official data file")
                        output.write(chunk)
                if not valid_transport(temporary):
                    raise ValueError(f"Official download for {name} is not a SAS transport file")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        sources.append({"name": name, "url": url, "codebook": BASE + name + ".htm",
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
        print(f"Verified official file: {name} ({path.stat().st_size} bytes)", flush=True)
    write_json(ROOT / "reports/data_manifest.json", {"dataset": config()["dataset"],
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "files": sources,
        "terms": "https://www.cdc.gov/nchs/policy/data-user-agreement.html"})


def validate_table(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    for column in required:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{name}: {column} must be numeric")
    ids = frame.SEQN
    if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any():
        raise ValueError(f"{name}: invalid participant identifier")
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate participant identifier")


def read_table(path: Path, name: str) -> pd.DataFrame:
    # pandas 2.2.3 corrupts SAS zeroes into 2**-260 (upstream #50670); use ReadStat.
    frame, _ = pyreadstat.read_xport(path)
    validate_table(frame, COLUMNS[name], name)
    if name == "P_GLU" and ((frame.WTSAFPRP > 0) & (frame.WTSAFPRP < 1)).any():
        raise ValueError("Unexpected positive fasting weight below 1; check XPT zero decoding")
    return frame


def prepare_frame(tables: dict[str, pd.DataFrame], settings: dict) -> tuple[pd.DataFrame, dict]:
    for name, columns in COLUMNS.items():
        validate_table(tables[name], columns, name)
    joined = tables["P_DEMO"][COLUMNS["P_DEMO"]].copy()
    counts = {name: len(frame) for name, frame in tables.items()}
    for name in COLUMNS:
        if name != "P_DEMO":
            joined = joined.merge(tables[name][COLUMNS[name]], on="SEQN", how="left", validate="one_to_one")
    frame = joined.rename(columns=RENAME)
    missing = {column: int(frame[column].isna().sum()) for column in [*FEATURES, "weight"]}
    systolic = frame[[f"BPXOSY{i}" for i in [1, 2, 3]]].to_numpy(float)
    diastolic = frame[[f"BPXODI{i}" for i in [1, 2, 3]]].to_numpy(float)
    valid = np.isfinite(systolic) & (systolic > 0) & (systolic <= 300)
    valid &= np.isnan(diastolic) | (np.isfinite(diastolic) & (diastolic > 0) & (systolic > diastolic))
    invalid_readings = int((np.isfinite(systolic) & ~valid).sum())
    frame["sbp_readings"] = valid.sum(axis=1)
    frame["target"] = pd.DataFrame(np.where(valid, systolic, np.nan)).mean(axis=1).to_numpy()
    attrition = [{"step": "demographics and one-to-one left joins", "remaining": len(frame), "excluded": 0}]

    def retain(mask: pd.Series, reason: str) -> None:
        nonlocal frame
        previous = len(frame)
        frame = frame.loc[mask].copy()
        attrition.append({"step": reason, "remaining": len(frame), "excluded": previous - len(frame)})

    retain(frame.age.between(settings["age_min"], settings["age_max"]), "age 18–79; exclude topcoded 80+")
    retain(np.isfinite(frame.weight) & (frame.weight > 0), "positive official fasting subsample weight")
    retain(frame.sex.isin([1, 2]), "documented sex category")
    retain(frame.sbp_readings >= settings["minimum_sbp_readings"], "at least two valid paired SBP readings")
    retain(np.isfinite(frame[FEATURES]).all(axis=1), "five observed finite features; no target or feature imputation")
    retain((frame[["bmi", "totalCholesterol", "glucose"]] > 0).all(axis=1), "positive measured BMI and SI laboratory values")
    retain(np.isfinite(frame[["stratum", "psu"]]).all(axis=1), "available survey design variables")
    frame["sex"] = frame.sex.map({1: 0., 2: 1.})
    frame["medication"] = frame.medication.map({1: "yes", 2: "no"}).fillna("unknown")
    columns = ["SEQN", *FEATURES, "target", "weight", "stratum", "psu", "medication", "sbp_readings"]
    frame = frame[columns].sort_values("SEQN").reset_index(drop=True)
    audit = {"raw_table_counts": counts, "merged_participants": len(joined), "missing_after_merge": missing,
             "attrition": attrition, "invalid_sbp_readings": invalid_readings,
             "target_definition": "Mean of >=2 valid oscillometric systolic readings at the same examination"}
    return frame, audit


def split_participants(ids: np.ndarray, seed: int, validation_fraction: float = .15, test_fraction: float = .15) -> np.ndarray:
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Participant identifiers must be unique before splitting")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction+test_fraction >= 1:
        raise ValueError("Invalid validation/test fractions")
    order = np.argsort(ids)
    permutation = np.random.default_rng(seed).permutation(len(ids))
    labels = np.full(len(ids), "train", dtype="<U10")
    test_size, validation_size = int(len(ids)*test_fraction), int(len(ids)*validation_fraction)
    labels[order[permutation[:test_size]]] = "test"
    labels[order[permutation[test_size:test_size+validation_size]]] = "validation"
    return labels


def prepare() -> None:
    guard_test_unopened()
    settings = config()
    manifest = ROOT / "reports/data_manifest.json"
    if not manifest.exists():
        raise ValueError("Run download first to record source provenance")
    from .common import read_json
    for item in read_json(manifest)["files"]:
        if sha256(ROOT / "data/raw" / f"{item['name']}.xpt") != item["sha256"]:
            raise ValueError("Raw file differs from the source manifest")
    tables = {name: read_table(ROOT / "data/raw" / f"{name}.xpt", name) for name in COLUMNS}
    frame, audit = prepare_frame(tables, settings)
    if len(frame) < 500:
        raise ValueError("Insufficient complete-case sample for the registered comparison")
    if not np.isclose(sum(settings[k] for k in ["train_fraction", "validation_fraction", "test_fraction"]), 1.):
        raise ValueError("Registered split fractions must sum to one")
    frame["split"] = split_participants(frame.SEQN.to_numpy(), settings["seed"], settings["validation_fraction"], settings["test_fraction"])
    destination = ROOT / "data/processed/dataset.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    frame[["SEQN", "split"]].to_csv(ROOT / "reports/split_manifest.csv", index=False)
    audit["final_participants"] = len(frame)
    fasting_weights = tables["P_GLU"].WTSAFPRP
    audit["xpt_reader"] = "pyreadstat " + pyreadstat.__version__
    audit["raw_fasting_weight_zero_count"] = int((fasting_weights == 0).sum())
    audit["raw_fasting_weight_positive_min"] = float(fasting_weights[fasting_weights > 0].min())
    audit["split_counts"] = frame.split.value_counts().to_dict()
    audit["missing_percent_after_merge"] = {k: round(v / len(tables["P_DEMO"]) * 100, 3) for k, v in audit["missing_after_merge"].items()}
    audit["complete_case_descriptive"] = frame[[*FEATURES, "target", "weight"]].describe().to_dict()
    audit["correlations_pearson_descriptive_only"] = frame[[*FEATURES, "target"]].corr().to_dict()
    audit["medication_counts"] = frame.medication.value_counts().to_dict()
    write_json(ROOT / "reports/data_audit.json", audit)
    write_json(ROOT / "reports/protocol.json", {"config": settings, "config_sha256": sha256(ROOT / "configs/experiment.json"),
        "dataset_sha256": sha256(destination), "source_manifest_sha256": sha256(manifest),
        "feature_order": FEATURES, "selection_locked_before_test": True})
    print(f"Prepared {len(frame)} real participants; split counts {audit['split_counts']}", flush=True)

