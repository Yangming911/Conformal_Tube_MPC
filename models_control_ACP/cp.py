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

from models_control.model_def import ACPCausalPedestrianPredictor
from models_control_ACP.train_ACP import load_from_csv, SequenceDataset



def load_model(model_path: str, device: torch.device, T: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1) -> ACPCausalPedestrianPredictor:
    model = ACPCausalPedestrianPredictor(p_dim=2, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def compute_errors_per_step(model: ACPCausalPedestrianPredictor, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        errors: [N, T] L2 errors per sample per step
        T: sequence length
    """
    all_errors = []
    all_speeds = []
    T_len = None
    with torch.no_grad():
        for past_p_seq, p_seq in loader:
            past_p_seq = past_p_seq.to(device)
            p_seq = p_seq.to(device)
            pred = model(past_p_seq)  # [B,T,2]
            err = torch.linalg.norm(pred - p_seq, dim=2)  # [B,T]

            batch_errors = err.cpu().numpy()  # [B,T]

            all_errors.append(batch_errors)
            if T_len is None:
                T_len = batch_errors.shape[1]

    errors = np.concatenate(all_errors, axis=0)
    return errors, T_len

def main():
    parser = argparse.ArgumentParser(description="Compute per-step conformal eta with 3 car_v bins")
    parser.add_argument('--model_path', type=str, default='assets_ACP/control_ped_model.pth', help='Trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of calibration sequences to generate')
    parser.add_argument('--T', type=int, default=10, help='Sequence length')
    parser.add_argument('--alpha', type=float, default=0.85, help='Quantile level in (0,1)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    parser.add_argument('--save_path', type=str, default='assets_ACP/cp_eta_ACP.csv', help='Where to save eta matrix as CSV')
    parser.add_argument('--save_edges_path', type=str, default='assets_ACP/cp_eta_edges_ACP.csv', help='Where to save bin edges as CSV')
    parser.add_argument('--save_errors_path', type=str, default='assets_ACP/cp_errors_ACP.npy', help='Where to save errors as NPY')
    parser.add_argument('--data_path', type=str, default='./assets_ACP/control_sequences_1025_cp.csv', help='Path to calibration data CSV')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    past_p_seq, p_seq = load_from_csv(args.data_path, args.T)
    ds = SequenceDataset(past_p_seq, p_seq)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
     # load model
    model = load_model(args.model_path, device, args.T)

    # compute errors
    errors, T_len = compute_errors_per_step(model, loader, device)
    # compute eta
    eta = np.zeros((T_len, 1), dtype=np.float32)
    for t in range(T_len):
        eta[t, 0] = np.quantile(errors[:, t], args.alpha)
    
    # save eta
    np.savetxt(args.save_path, eta, delimiter=',', fmt='%.6f')
    # save errors
    np.save(args.save_errors_path, errors)
    print(f"Eta saved to {args.save_path}")
    print(f"Errors saved to {args.save_errors_path}")

if __name__ == '__main__':
    main()


