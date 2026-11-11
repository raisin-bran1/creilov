"""Compatibility CLI for the reusable predictor report generator."""

import argparse
from pathlib import Path
from creilov.predictor_report import build_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(build_table(parser.parse_args().run_dir))
