#!/usr/bin/env python3
"""
批量调用 evalcbf_multi.py，针对多组行人数量运行评估：1, 3, 5, 7, 9。

用法示例：
  - 快速模式（每组100样本）:
      python tools/run_evalcbf_multi_batch.py --quick
  - 常规模式（使用 evalcbf_multi.py 的默认 sample_num）:
      python tools/run_evalcbf_multi_batch.py
  - 自定义gamma参数:
      python tools/run_evalcbf_multi_batch.py --gamma 0.5
  - 自定义行人数量列表:
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


