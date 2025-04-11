import argparse
import os
import pickle

from models.conformal_grid import build_conformal_grid

def main():
    parser = argparse.ArgumentParser(description="Generate conformal prediction grid and save it.")
    parser.add_argument('--alpha', type=float, default=0.9, help='Quantile level for conformal region')
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples for CP calibration')
    parser.add_argument('--save_path', type=str, default='assets/conformal_grid.pkl', help='Path to save the grid')
    args = parser.parse_args()

    print(f"Generating CP grid with alpha={args.alpha}, samples={args.num_samples}...")
    grid = build_conformal_grid(alpha=args.alpha, num_samples=args.num_samples)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(grid, f)

    print(f"Conformal grid saved to {args.save_path}")
    
if __name__ == '__main__':
    main()
