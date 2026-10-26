from __future__ import annotations

import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits

from .data import download, prepare
from .training import train, calibrate, evaluate
from .bundle import Bundle, export


def main():
    parser = argparse.ArgumentParser(description="Independent D1 HbA1c screening experiment")
    parser.add_argument("command", choices=["download", "prepare", "train", "calibrate", "evaluate", "export", "verify"])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--run", type=Path, default=Path("runs/d1-r2"))
    parser.add_argument("--config", type=Path, default=Path("configs/d1_hba1c_screen.json"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/d1"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "download": download(args.raw)
        elif args.command == "prepare": prepare(args.raw, args.run, args.config)
        elif args.command == "export": export(args.run, args.bundle)
        elif args.command == "verify":
            bundle = Bundle(args.bundle)
            print("Verified bundle and synthetic parity:", bundle.metadata["modelVersion"])
        else: {"train": train, "calibrate": calibrate, "evaluate": evaluate}[args.command](args.run)


if __name__ == "__main__":
    main()
