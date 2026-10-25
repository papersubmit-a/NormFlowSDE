"""Compatibility entry point for the maintained mass-spring experiment.

The former copy of the training driver diverged in its likelihood, refinement
flags and holdout handling. Keep one implementation in the experiments package.
"""
from pathlib import Path
import runpy
import sys


def main():
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    runpy.run_module("experiments.mass_spring_damper.train", run_name="__main__")


if __name__ == "__main__":
    main()
