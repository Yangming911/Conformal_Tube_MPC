#!/usr/bin/env python3
"""
Compute conformal regions for sequence model per time step using constant-speed calibration data.

For each time step t=1..T and for 3 bins of car_v, compute the alpha-quantile
of L2 errors ||y_t - y_pred_t||, where y_t is the true pedestrian position.

Only bin along car_v (3 bins). All other inputs are not binned.
Output: eta matrix of shape [T, 3] saved to assets, plus a pickle with metadata.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root on path BEFORE importing project modules
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models_control.model_def import CausalPedestrianPredictor
from tools.collect_control_sequences import collect_dataset as collect_constant_speed_dataset
from models_control.train import load_from_csv


class SequenceDataset(Dataset):
    def __init__(self, u: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, p_ped_seq: np.ndarray):
        self.u = torch.from_numpy(u).float()
        self.p_veh0 = torch.from_numpy(p_veh0).float()
        self.p_ped0 = torch.from_numpy(p_ped0).float()
        self.p_ped_seq = torch.from_numpy(p_ped_seq).float()

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx: int):
        return self.u[idx], self.p_veh0[idx], self.p_ped0[idx], self.p_ped_seq[idx]


def load_model(model_path: str, device: torch.device, T: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1) -> CausalPedestrianPredictor:
    model = CausalPedestrianPredictor(u_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    scaler = None
    if 'u_scaler' in checkpoint.keys():
        scaler = {
            'u_scaler': checkpoint['u_scaler'],
            'p_veh0_scaler': checkpoint['p_veh0_scaler'],
            'p_ped0_scaler': checkpoint['p_ped0_scaler'],
            'p_seq_scaler': checkpoint['p_seq_scaler'],
        }
    return model, scaler


def compute_errors_per_step(model: CausalPedestrianPredictor, loader: DataLoader, device: torch.device, scaler=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        errors: [N, T] L2 errors per sample per step
        speeds: [N] constant speed per sample (assume u is constant over T)
        T: sequence length
    """
    all_errors = []
    all_speeds = []
    T_len = None
    with torch.no_grad():
        for u, p_veh0, p_ped0, p_seq in loader:
            u = u.to(device)
            p_veh0 = p_veh0.to(device)
            p_ped0 = p_ped0.to(device)
            p_seq = p_seq.to(device)

            pred = model(u, p_veh0, p_ped0)  # [B,T,2]
            if scaler is not None:
                # Inverse transform predictions and ground truth
                p_seq_reshape = p_seq.reshape(-1, 2).detach().cpu().numpy()
                pred_reshape = pred.reshape(-1, 2).detach().cpu().numpy()
                p_seq_reshape = scaler['p_seq_scaler'].inverse_transform(p_seq_reshape)
                pred_reshape = scaler['p_seq_scaler'].inverse_transform(pred_reshape)
                p_seq = p_seq_reshape.reshape(p_seq.shape)
                pred = pred_reshape.reshape(pred.shape)
                u_reshape = u.reshape(-1, 1).detach().cpu().numpy()
                u_reshape = scaler['u_scaler'].inverse_transform(u_reshape)
                u = u_reshape.reshape(u.shape)
                u = torch.from_numpy(u).to(device)
            err = torch.linalg.norm(pred - p_seq, dim=2)  # [B,T]

            batch_errors = err.cpu().numpy()  # [B,T]
            speeds = u[:, 0, 0].cpu().numpy()  # [B]

            all_errors.append(batch_errors)
            all_speeds.append(speeds)
            if T_len is None:
                T_len = batch_errors.shape[1]

    errors = np.concatenate(all_errors, axis=0)
    speeds = np.concatenate(all_speeds, axis=0)
    return errors, speeds, T_len


def bin_indices_for_speed(speeds: np.ndarray) -> np.ndarray:
    """
    3 speed bins across [0, 15]: [0,5), [5,10), [10,15+]
    Returns integer indices in {0,1,2} per sample
    """
    edges = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float32)
    # clip speeds into range for safety
    s = np.clip(speeds, 0.0, 15.0 - 1e-6)
    idx = np.digitize(s, edges) - 1
    idx = np.clip(idx, 0, 2)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Compute per-step conformal eta with 3 car_v bins")
    parser.add_argument('--model_path', type=str, default='assets/control_ped_model.pth', help='Trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of calibration sequences to generate')
    parser.add_argument('--T', type=int, default=10, help='Sequence length')
    parser.add_argument('--alpha', type=float, default=0.85, help='Quantile level in (0,1)')
    parser.add_argument('--num_bins', type=int, default=5, help='Number of speed bins over [0,15]')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    parser.add_argument('--min_count', type=int, default=20, help='Minimum samples per bin; fallback if fewer')
    parser.add_argument('--fallback_eta', type=float, default=0.5, help='Fallback eta when bin underpopulated')
    parser.add_argument('--save_path', type=str, default='assets/cp_eta_test.csv', help='Where to save eta matrix as CSV')
    parser.add_argument('--save_edges_path', type=str, default='assets/cp_eta_edges_test.csv', help='Where to save bin edges as CSV')
    parser.add_argument('--p2p', type=bool, default=False, help='Whether to use p2p model')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect constant-speed calibration dataset
    print(f"Collecting calibration dataset: episodes={args.episodes}, T={args.T}")
    u, p_veh0, p_ped0, p_seq = collect_constant_speed_dataset(args.episodes, args.T, seed=2025, p2p=args.p2p)

    # Load model
    print(f"Loading model from {args.model_path}")
    model, scaler = load_model(args.model_path, device=device, T=args.T)

    # scaler data
    if scaler is not None:
        u = scaler['u_scaler'].transform(u.reshape(-1, 1)).reshape(u.shape)
        p_veh0 = scaler['p_veh0_scaler'].transform(p_veh0)
        p_ped0 = scaler['p_ped0_scaler'].transform(p_ped0)
        p_seq = scaler['p_seq_scaler'].transform(p_seq.reshape(-1, 2)).reshape(p_seq.shape)

    # DataLoader
    ds = SequenceDataset(u, p_veh0, p_ped0, p_seq)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Compute errors and speed bins
    print("Running model to compute per-step errors...")
    errors, speeds, T_len = compute_errors_per_step(model, loader, device, scaler)
    assert T_len == args.T, f"Sequence length mismatch: got {T_len}, expected {args.T}"
    # Build dynamic speed bins
    num_bins = max(1, int(args.num_bins))
    edges = np.linspace(0.0, 15.0, num_bins + 1).astype(np.float32)  # inclusive of 0 and 15
    # Digitize speeds into [0,num_bins-1]
    s = np.clip(speeds, edges[0], edges[-1] - 1e-6)
    v_bins = np.digitize(s, edges) - 1
    v_bins = np.clip(v_bins, 0, num_bins - 1)

    # Compute per-step quantiles per speed bin
    eta = np.zeros((args.T, num_bins), dtype=np.float32)
    counts = np.zeros((num_bins,), dtype=np.int64)
    for b in range(num_bins):
        mask = v_bins == b
        counts[b] = int(mask.sum())
        for t in range(args.T):
            if counts[b] < args.min_count:
                eta[t, b] = float(args.fallback_eta)
            else:
                eta[t, b] = float(np.quantile(errors[mask, t], args.alpha))

    # Save results to CSV with columns: t, bin0, bin1, bin2
    import pandas as pd
    # Build dynamic column names with edges
    col_names = []
    for b in range(num_bins):
        left = edges[b]
        right = edges[b+1]
        col_names.append(f"bin{b}_[{left:.3f},{right:.3f})")
    df = pd.DataFrame(eta, columns=col_names)
    df.insert(0, "t", np.arange(1, args.T + 1))
    df.to_csv(args.save_path, index=False)
    # Save edges separately for robust parsing
    pd.DataFrame({"edges": edges}).to_csv(args.save_edges_path, index=False)
    print(f"Saved CP eta CSV to {args.save_path} with {num_bins} bins; edges -> {args.save_edges_path}")


if __name__ == '__main__':
    main()


