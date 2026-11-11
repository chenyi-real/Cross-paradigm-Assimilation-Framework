"""Run a legacy SOC entry point with explicit project/config import paths."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["nasa", "cs2", "mit", "transfer", "distill"])
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    entries = {"nasa": "main.py", "cs2": "main_cs2.py", "mit": "main_mit.py",
               "transfer": "main_transfer_udds.py", "distill": "main_distill_udds.py"}
    for directory in (root, root / "configs", root / "train_script"):
        sys.path.insert(0, str(directory))
    entry = root / "train_script" / entries[args.experiment]
    sys.argv = [str(entry), *remaining]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
