import argparse
from simulation.run_simulation import run_visual_sim, run_batch_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='model', choices=['model'])
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--T', type=int, default=10000)
    args = parser.parse_args()

    if args.display:
        run_visual_sim()
    else:
        run_batch_sim(n=args.T)
