from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split

from ..common import FEATURES, read_json, sha256, write_json
from ..data import BASE, valid_transport, validate_table

TASK = "D1-HBA1C-SCREEN-v1"
COLUMNS = {
    "P_DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "SDMVSTRA", "SDMVPSU"],
    "P_BMX": ["SEQN", "BMXBMI"],
    "P_TCHOL": ["SEQN", "LBDTCSI"],
    "P_GLU": ["SEQN", "LBDGLUSI", "WTSAFPRP"],
    "P_GHB": ["SEQN", "LBXGH"],
    "P_DIQ": ["SEQN", "DIQ010", "DIQ050", "DIQ070", "DIQ160"],
}


def fingerprints() -> dict:
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def guard_unopened(run: Path) -> None:
    if (run / "test_opened.json").exists():
        raise RuntimeError("Final test already opened; preserve this run and use a new protocol/workspace")


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in COLUMNS:
        path = raw / f"{name}.xpt"
        if not path.exists():
            temporary = path.with_suffix(".part")
            try:
                with urllib.request.urlopen(BASE + path.name, timeout=60) as response:
                    data = response.read(30_000_001)
                if len(data) > 30_000_000:
                    raise ValueError("Source exceeds expected size")
                temporary.write_bytes(data)
                if not valid_transport(temporary):
                    raise ValueError("Source is not SAS XPT")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not valid_transport(path):
            raise ValueError("Invalid local XPT source")
        sources.append({"name": name, "url": BASE + path.name, "codebook": BASE + name + ".htm",
                        "sha256": sha256(path), "bytes": path.stat().st_size})
    write_json(raw / "manifest.json", {"files": sources, "dataset": "NHANES 2017–March 2020",
               "terms": "https://www.cdc.gov/nchs/policy/data-user-agreement.html"})


def cohort(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    for name, columns in COLUMNS.items():
        validate_table(tables[name], columns, name)
    joined = tables["P_DEMO"][COLUMNS["P_DEMO"]].copy()
    for name in COLUMNS:
        if name != "P_DEMO":
            joined = joined.merge(tables[name][COLUMNS[name]], on="SEQN", how="left", validate="one_to_one")
    frame = joined.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "BMXBMI": "bmi",
        "LBDTCSI": "totalCholesterol", "LBDGLUSI": "glucose", "WTSAFPRP": "weight",
        "SDMVSTRA": "stratum", "SDMVPSU": "psu"})
    audit = {"source_counts": {k: len(v) for k, v in tables.items()}, "attrition": [],
             "missing_before_exclusions": frame.isna().sum().astype(int).to_dict()}

    def retain(mask, reason):
        nonlocal frame
        before = len(frame)
        frame = frame.loc[mask].copy()
        audit["attrition"].append({"step": reason, "excluded": before-len(frame), "remaining": len(frame)})

    retain(frame.age.between(18, 79) & frame.sex.isin([1, 2]), "age 18–79, documented sex")
    retain(np.isfinite(frame.weight) & (frame.weight >= 1), "positive fasting weight; exclude SAS zero decoding artifacts")
    # Male RIDEXPRG is structurally inapplicable; missing female status stays unknown.
    retain((frame.sex == 1) | ((frame.sex == 2) & (frame.RIDEXPRG == 2)), "male or observed not-pregnant female")
    retain(frame.DIQ010 == 2, "explicit self-report of no doctor-diagnosed diabetes; exclude borderline/unknown")
    # DIQ065 skips pills when DIQ010=2 and DIQ160=2. This is NOT verified absence of medication.
    skip_pills = frame.DIQ070.isna() & (frame.DIQ160 == 2)
    retain((frame.DIQ070 == 2) | skip_pills, "no reported pills or documented DIQ065 skip; unknown responses excluded")
    retain(frame.DIQ050.isna() | (frame.DIQ050 == 2), "exclude any reported insulin; structural skip remains unassessed")
    retain(np.isfinite(frame[FEATURES]).all(axis=1) & (frame[["bmi", "totalCholesterol", "glucose"]] > 0).all(axis=1), "five finite complete-case inputs")
    retain(np.isfinite(frame.LBXGH) & (frame.LBXGH > 0), "observed positive HbA1c; never impute label")
    retain(np.isfinite(frame[["stratum", "psu"]]).all(axis=1), "survey design available")
    frame["target"] = (frame.LBXGH >= 6.5).astype(int)
    frame["sex"] = frame.sex.map({1: 0., 2: 1.})
    frame = frame[["SEQN", *FEATURES, "target", "weight", "stratum", "psu"]].sort_values("SEQN").reset_index(drop=True)
    audit.update(n=len(frame), positives=int(frame.target.sum()), negatives=int((frame.target == 0).sum()),
        target="single measured HbA1c >=6.5%; not confirmed diabetes",
        limitations=["Insulin question skipped in undiagnosed respondents: treatment-naive status not established",
            "Pills skipped per DIQ065 retained only with explicit DIQ160=2; skip is not observed no treatment",
            "Unknown female pregnancy status excluded, substantially restricting female age coverage",
            "Complete-case fasting selection; no Kazakhstan or clinical validation"])
    return frame, audit


def split(frame: pd.DataFrame, seed: int) -> pd.Series:
    if frame.SEQN.duplicated().any() or not set(frame.target.unique()) == {0, 1}:
        raise ValueError("Unique participants and both classes required")
    ids = frame.index.to_numpy()
    train, test = train_test_split(ids, test_size=.2, stratify=frame.target, random_state=seed)
    train, validation = train_test_split(train, test_size=.25, stratify=frame.loc[train, "target"], random_state=seed)
    calibration, threshold = train_test_split(validation, test_size=.5, stratify=frame.loc[validation, "target"], random_state=seed)
    labels = pd.Series(index=frame.index, dtype="object")
    for name, index in [("train", train), ("calibration", calibration), ("threshold", threshold), ("test", test)]:
        labels.loc[index] = name
    return labels


def prepare(raw: Path, run: Path, config: Path) -> None:
    guard_unopened(run)
    if (run / "protocol.json").exists():
        raise RuntimeError("Run already prepared; do not overwrite a registered split")
    manifest = read_json(raw / "manifest.json")
    tables = {}
    for source in manifest["files"]:
        path = raw / (source["name"] + ".xpt")
        if sha256(path) != source["sha256"]:
            raise ValueError("Source hash mismatch")
        tables[source["name"]], _ = pyreadstat.read_xport(path)
    frame, audit = cohort(tables)
    settings = read_json(config)
    frame["split"] = split(frame, settings["seed"])
    counts = {name: {"n": len(part), "positives": int(part.target.sum()), "negatives": int((part.target == 0).sum())}
              for name, part in frame.groupby("split")}
    if min(v["positives"] for v in counts.values()) < settings["minimum_partition_positives"]:
        raise ValueError("Insufficient positive cases for the registered protocol; revise before training")
    run.mkdir(parents=True, exist_ok=True)
    frame.to_csv(run / "dataset.csv", index=False)
    frame[["SEQN", "split"]].to_csv(run / "split_manifest.csv", index=False)
    audit["partitions"] = counts
    audit["precision_caution"] = "Partition class counts are feasibility checks, not proof of precise calibration or sensitivity"
    write_json(run / "cohort_audit.json", audit)
    write_json(run / "data_manifest.json", manifest)
    write_json(run / "protocol.json", {"task": TASK, "config": settings, "config_sha256": sha256(config),
        "dataset_sha256": sha256(run / "dataset.csv"), "sources": fingerprints(),
        "eligibility_revision": "D1-r2: documented DIQ065 skips retained as unassessed, not medication-free",
        "feature_order": FEATURES})
    print(json.dumps({"cohort": audit["n"], "partitions": counts}))


def dataset(run: Path) -> tuple[pd.DataFrame, dict]:
    protocol = read_json(run / "protocol.json")
    if sha256(run / "dataset.csv") != protocol["dataset_sha256"] or fingerprints() != protocol["sources"]:
        raise ValueError("Dataset or training source changed after registration")
    frame = pd.read_csv(run / "dataset.csv")
    if frame.SEQN.duplicated().any() or not set(frame.split.unique()) == {"train", "calibration", "threshold", "test"}:
        raise ValueError("Invalid participant partitions")
    return frame, protocol
