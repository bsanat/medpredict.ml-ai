from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MedPredict reproducible NHANES research pipeline")
    parser.add_argument("command", choices=["download", "prepare", "train", "evaluate", "export", "verify"])
    parser.add_argument("--workspace", type=Path, help="Separate output workspace for an exact reproduction, preserving the original frozen experiment")
    arguments = parser.parse_args()
    if arguments.workspace:
        from . import common
        target = arguments.workspace.resolve()
        (target / "configs").mkdir(parents=True, exist_ok=True)
        if not (target / "configs/experiment.json").exists():
            shutil.copyfile(common.SOURCE_ROOT / "configs/experiment.json", target / "configs/experiment.json")
        common.ROOT = target
    # Small tabular workload: avoid oversubscribing BLAS/OpenMP and make timing repeatable.
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(name, "1")
    if arguments.command in ["download", "prepare"]:
        from . import data
        getattr(data, arguments.command)()
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            import torch
            torch.set_num_threads(1)
            if arguments.command == "train":
                from .training import train
                train()
            elif arguments.command == "evaluate":
                from .evaluation import evaluate
                evaluate()
            else:
                from . import export
                getattr(export, arguments.command)()


if __name__ == "__main__":
    main()
