#!/usr/bin/env python3
"""
Identify which architecture a saved model file corresponds to and compare parameter counts.

This script will:
- Build both architectures from models.model_def
- Print parameter counts for each architecture
- Attempt to load state_dict from each candidate .pth into each architecture (strict=True/False)
- Report which architecture matches best_model.pth (and other provided files)

Usage:
  python tools/identify_best_model.py \
    --files assets/best_model.pth assets/walker_speed_predictor_new.pth assets/walker_speed_predictor_v2_fixed.pth

If --files is omitted, it defaults to the three files above (if present).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

# Ensure project root on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model_def import WalkerSpeedPredictor, WalkerSpeedPredictorV2  # noqa: E402


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_load(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[bool, int, int]:
    """Try strict load first; if fails, try non-strict and report missing/unexpected counts.

    Returns: (strict_ok, missing, unexpected)
    """
    try:
        model.load_state_dict(state, strict=True)
        return True, 0, 0
    except Exception:
        missing, unexpected = model.load_state_dict(state, strict=False)
        return False, len(missing), len(unexpected)


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    # Handle various save formats: direct state_dict, {'model_state_dict': ...}, {'state_dict': ...}
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        # If it looks like a plain state_dict (tensor values)
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore[return-value]
    raise ValueError(f"Unrecognized checkpoint format for {path}")


def main():
    parser = argparse.ArgumentParser(description="Identify best_model architecture and compare params")
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "assets/best_model.pth",
            "assets/walker_speed_predictor_new.pth",
            "assets/walker_speed_predictor_v2_fixed.pth",
        ],
        help="List of checkpoint files to test",
    )
    args = parser.parse_args()

    # Build reference architectures
    archs = {
        "WalkerSpeedPredictor": WalkerSpeedPredictor(input_dim=7, hidden_dims=[128, 128, 64], output_dim=2, dropout_rate=0.1),
        "WalkerSpeedPredictorV2": WalkerSpeedPredictorV2(input_dim=7, hidden_dim=128, output_dim=2, num_layers=4, dropout_rate=0.1),
    }

    print("Architectures and parameter counts:")
    for name, model in archs.items():
        print(f"- {name}: {count_parameters(model):,} parameters")
    print()

    files = [f for f in args.files if os.path.exists(f)]
    if not files:
        print("No checkpoint files found. Please pass --files with valid paths.")
        sys.exit(1)

    for ckpt in files:
        print(f"Checking: {ckpt}")
        try:
            state = load_state_dict(ckpt)
        except Exception as e:
            print(f"  ! Failed to load: {e}")
            continue

        # Basic stats
        total_tensors = len(state)
        total_params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
        print(f"  State tensors: {total_tensors}, total params in checkpoint: {total_params:,}")

        # Try match with each architecture
        best_match = None
        for name, model in archs.items():
            strict_ok, missing, unexpected = try_load(model, state)
            status = "STRICT MATCH" if strict_ok else f"PARTIAL (missing={missing}, unexpected={unexpected})"
            print(f"  -> {name}: {status}")
            if strict_ok and best_match is None:
                best_match = name

        if best_match is None:
            print("  => No strict match. The checkpoint may require a different config or is incompatible.")
        else:
            print(f"  => Identified as: {best_match}")
        print()


if __name__ == "__main__":
    main()





