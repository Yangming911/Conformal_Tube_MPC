#!/usr/bin/env python3
"""
Batch call evalcbf_multi.py, run evaluation for multiple pedestrian counts: 1, 3, 5, 7, 9.

Usage examples:
  - Quick mode (100 samples per group):
      python tools/run_evalcbf_multi_batch.py --quick
  - Regular mode (use evalcbf_multi.py default sample_num):
      python tools/run_evalcbf_multi_batch.py
  - Custom gamma parameter:
      python tools/run_evalcbf_multi_batch.py --gamma 0.5
  - Custom pedestrian count list:
      python tools/run_evalcbf_multi_batch.py --nums 1,2,4,8
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for evalcbf_multi.py")
    parser.add_argument(
        "--nums",
        type=str,
        default="1,3,5,7,9",
        help="Comma-separated list of pedestrian counts to evaluate (default: 1,3,5,7,9)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Enable quick mode (passes --quick to evalcbf_multi.py)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="CBF gamma parameter (default: 1.0)",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra arguments passed verbatim to evalcbf_multi.py, e.g. --extra '--cp_alpha 0.85'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure we run from project root so relative paths in evalcbf_multi.py work
    project_root = Path(__file__).resolve().parents[1]
    eval_script = project_root / "evalcbf_multi.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Cannot find eval script: {eval_script}")

    # Parse nums list
    try:
        nums = [int(x.strip()) for x in args.nums.split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--nums must be a comma-separated list of integers, e.g. 1,3,5")

    # Build constant parts of the command
    python_exe = sys.executable or "python"

    for n in nums:
        cmd = [python_exe, str(eval_script), "--num_pedestrians", str(n), "--gamma", str(args.gamma)]
        if args.quick:
            cmd.append("--quick")
        if args.extra:
            # naive split by space to allow simple passthrough; for complex quoting, prefer calling directly
            cmd.extend(args.extra.split())

        print("=" * 80)
        print(f"Running evalcbf_multi.py with num_pedestrians={n}, gamma={args.gamma} {'[quick]' if args.quick else ''}")
        print("Command:", " ".join(cmd))
        print("=" * 80)

        # Run in project root so that logs/ paths resolve correctly
        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            raise SystemExit(
                f"Run failed for num_pedestrians={n} with exit code {result.returncode}"
            )


if __name__ == "__main__":
    main()


