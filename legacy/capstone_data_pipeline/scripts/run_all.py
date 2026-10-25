"""Run all pipeline steps in order. Stops on the first failure."""
import subprocess
import sys
import os

STEPS = [
    "01_load_trajectories.py",
    "02_load_swe_smith.py",
    "03_load_swe_bench_experiments.py",
    "04_load_agenticvbench.py",
    "05_combine_datasets.py",
    # run_harbor_benchmark.py is NOT included here — it costs real API
    # spend and requires --family/--agent/--model args, so run it manually.
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    for step in STEPS:
        path = os.path.join(SCRIPT_DIR, step)
        print(f"\n=== running {step} ===")
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"\n{step} failed — stopping. Fix the issue and re-run "
                  f"(previously completed steps' output is cached in data/raw/).")
            sys.exit(result.returncode)
    print("\nAll steps completed. Combined dataset in data/processed/.")
