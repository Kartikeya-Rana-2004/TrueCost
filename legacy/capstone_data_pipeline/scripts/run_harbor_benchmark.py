"""OPTIONAL — run AgenticVBench yourself via Harbor to get real per-task
trajectory + cost data (not published anywhere; must be generated).

Costs real API spend: roughly $0.10-$2 per task depending on model, and
~10 min wall clock per task. A full 100-task sweep with a mid-tier model is
order-of-magnitude $10-200. Do not run this unattended without a budget cap
in mind.

This is a thin wrapper around the `avb` CLI shipped in the cloned repo
(scripts/04_load_agenticvbench.py clones it to
data/raw/agenticvbench_manual/agentic-vbench-repo/). Requires the relevant
model API key exported (e.g. ANTHROPIC_API_KEY) and, for a Modal-parallel
run, MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/run_harbor_benchmark.py --family repair --agent claude-code \
        --model anthropic/claude-sonnet-4-6 --max-parallel 5
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENTICVBENCH_DIR
from scripts.utils import log, fail

REPO_DIR = os.path.join(AGENTICVBENCH_DIR, "agentic-vbench-repo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True,
                         choices=["repair", "assembly", "sequencing", "repurpose"])
    parser.add_argument("--agent", required=True,
                         help="e.g. claude-code, codex, gemini-cli, opencode")
    parser.add_argument("--model", required=True,
                         help="e.g. anthropic/claude-sonnet-4-6")
    parser.add_argument("--env", default="docker", help="executor: docker, modal, e2b, ...")
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(REPO_DIR):
        fail(f"{REPO_DIR} not found — run scripts/04_load_agenticvbench.py first "
             "to clone the task repo.")

    install_script = os.path.join(REPO_DIR, "scripts", "install-harbor.sh")
    if not os.path.exists(os.path.join(REPO_DIR, ".venv")):
        log("installing Harbor ...")
        subprocess.run(["bash", install_script], cwd=REPO_DIR, check=True)

    log(f"running family={args.family} agent={args.agent} model={args.model} "
        f"env={args.env} max-parallel={args.max_parallel} — this costs real API spend")

    cmd = [
        "./avb", "rollout",
        "--family", f"agentic_vbench_{args.family}",
        "--agent", args.agent,
        "--model", args.model,
        "--env", args.env,
        "--max-parallel", str(args.max_parallel),
    ]
    subprocess.run(cmd, cwd=REPO_DIR, check=True)

    log(f"done. Per-trial trajectories/rewards under {REPO_DIR}/jobs/, "
        f"summary at {REPO_DIR}/logs/rollout-results.tsv")


if __name__ == "__main__":
    main()
