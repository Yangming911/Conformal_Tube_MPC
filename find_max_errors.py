#!/usr/bin/env python3
"""
================================================================================
CALIBRATION DATASET ERROR ANALYSIS V1
================================================================================
Dataset size: 148,006 samples

X-direction errors (walker_vx):
----------------------------------------
  Max error:     0.691134
  Min error:     0.000000
  Mean error:    0.080082
  Std error:     0.060973
  50th percentile: 0.067469
  75th percentile: 0.115608
  85th percentile: 0.144394
  90th percentile: 0.165113
  95th percentile: 0.197097
  99th percentile: 0.260602
  99.9th percentile: 0.342552

Y-direction errors (walker_vy):
----------------------------------------
  Max error:     2.326408
  Min error:     0.000005
  Mean error:    0.399069
  Std error:     0.296979
  50th percentile: 0.340333
  75th percentile: 0.577095
  85th percentile: 0.719968
  90th percentile: 0.820673
  95th percentile: 0.973102
  99th percentile: 1.244136
  99.9th percentile: 1.587395
  
  
  
================================================================================
CALIBRATION DATASET ERROR ANALYSIS V2
================================================================================
Dataset size: 148,117 samples

X-direction errors (walker_vx):
----------------------------------------
  Max error:     0.666312
  Min error:     0.000001
  Mean error:    0.080410
  Std error:     0.061049
  50th percentile: 0.067946
  75th percentile: 0.115841
  85th percentile: 0.144635
  90th percentile: 0.165629
  95th percentile: 0.198014
  99th percentile: 0.260089
  99.9th percentile: 0.339288

Y-direction errors (walker_vy):
----------------------------------------
  Max error:     2.463981
  Min error:     0.000003
  Mean error:    0.398192
  Std error:     0.296394
  50th percentile: 0.339933
  75th percentile: 0.576356
  85th percentile: 0.717887
  90th percentile: 0.822073
  95th percentile: 0.971790
  99th percentile: 1.235191
  99.9th percentile: 1.606853
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.predictor import WalkerActionPredictor
from models.conformal_grid import _collect_dataframe
import utils.constants as C

def find_max_errors(calib_csv_path=None, model_path="assets/best_model.pth"):
    """
    查找calibration数据集中x和y方向的最大误差分数
    
    Args:
        calib_csv_path: 校准数据CSV路径，如果为None则重新收集
        model_path: 模型路径
    
    Returns:
        dict: 包含最大误差统计信息的字典
    """
    print("Loading calibration data...")
    
    # 加载校准数据
    if calib_csv_path is not None:
        df = pd.read_csv(calib_csv_path)
        print(f"Loaded calibration data from {calib_csv_path}")
    else:
        df = _collect_dataframe()
        print("Collected fresh calibration data")
    
    print(f"Calibration dataset shape: {df.shape}")
    
    # 加载预测模型
    print("Loading prediction model...")
    predictor = WalkerActionPredictor(model_path=model_path, device="cpu")
    
    # 计算模型预测
    print("Computing model predictions...")
    preds_vx = []
    preds_vy = []
    
    for _, row in df.iterrows():
        vx_pred, vy_pred = predictor.predict(
            car_x=row["car_x"],
            car_y=row["car_y"],
            car_v=row["car_v"],
            walker_x=row["walker_x"],
            walker_y=row["walker_y"],
            walker_vx=row["walker_vx"],
            walker_vy=row["walker_vy"],
        )
        preds_vx.append(vx_pred)
        preds_vy.append(vy_pred)
    
    preds_vx = np.asarray(preds_vx, dtype=np.float32)
    preds_vy = np.asarray(preds_vy, dtype=np.float32)
    
    # 真实值
    true_next_vx = df["next_walker_vx"].to_numpy(dtype=np.float32)
    true_next_vy = df["next_walker_vy"].to_numpy(dtype=np.float32)
    
    # 计算误差分数（绝对误差）
    score_x = np.abs(preds_vx - true_next_vx)
    score_y = np.abs(preds_vy - true_next_vy)
    
    # save score_x and score_y for further analysis
    np.save('assets_calibration_errors_x.npy', score_x)
    np.save('assets_calibration_errors_y.npy', score_y)
    # 统计信息
    stats = {
        "dataset_size": len(df),
        "x_errors": {
            "max": float(np.max(score_x)),
            "min": float(np.min(score_x)),
            "mean": float(np.mean(score_x)),
            "std": float(np.std(score_x)),
            "percentile_50": float(np.percentile(score_x, 50)),
            "percentile_75": float(np.percentile(score_x, 75)),
            "percentile_85": float(np.percentile(score_x, 85)),
            "percentile_90" : float(np.percentile(score_x, 90)),
            "percentile_95": float(np.percentile(score_x, 95)),
            "percentile_99": float(np.percentile(score_x, 99)),
            "percentile_99_9": float(np.percentile(score_x, 99.9)),
        },
        "y_errors": {
            "max": float(np.max(score_y)),
            "min": float(np.min(score_y)),
            "mean": float(np.mean(score_y)),
            "std": float(np.std(score_y)),
            "percentile_50": float(np.percentile(score_y, 50)),
            "percentile_75": float(np.percentile(score_y, 75)),
            "percentile_85": float(np.percentile(score_y, 85)),
            "percentile_90" : float(np.percentile(score_y, 90)),
            "percentile_95": float(np.percentile(score_y, 95)),
            "percentile_99": float(np.percentile(score_y, 99)),
            "percentile_99_9": float(np.percentile(score_y, 99.9)),
        }
    }
    
    return stats, score_x, score_y

def print_error_analysis(stats):
    """打印误差分析结果"""
    print("\n" + "="*80)
    print("CALIBRATION DATASET ERROR ANALYSIS")
    print("="*80)
    print(f"Dataset size: {stats['dataset_size']:,} samples")
    
    print("\nX-direction errors (walker_vx):")
    print("-" * 40)
    x_stats = stats['x_errors']
    print(f"  Max error:     {x_stats['max']:.6f}")
    print(f"  Min error:     {x_stats['min']:.6f}")
    print(f"  Mean error:    {x_stats['mean']:.6f}")
    print(f"  Std error:     {x_stats['std']:.6f}")
    print(f"  50th percentile: {x_stats['percentile_50']:.6f}")
    print(f"  75th percentile: {x_stats['percentile_75']:.6f}")
    print(f"  85th percentile: {x_stats['percentile_85']:.6f}")
    print(f"  90th percentile: {x_stats['percentile_90']:.6f}")
    print(f"  95th percentile: {x_stats['percentile_95']:.6f}")
    print(f"  99th percentile: {x_stats['percentile_99']:.6f}")
    print(f"  99.9th percentile: {x_stats['percentile_99_9']:.6f}")
    
    print("\nY-direction errors (walker_vy):")
    print("-" * 40)
    y_stats = stats['y_errors']
    print(f"  Max error:     {y_stats['max']:.6f}")
    print(f"  Min error:     {y_stats['min']:.6f}")
    print(f"  Mean error:    {y_stats['mean']:.6f}")
    print(f"  Std error:     {y_stats['std']:.6f}")
    print(f"  50th percentile: {y_stats['percentile_50']:.6f}")
    print(f"  75th percentile: {y_stats['percentile_75']:.6f}")
    print(f"  85th percentile: {y_stats['percentile_85']:.6f}")
    print(f"  90th percentile: {y_stats['percentile_90']:.6f}")
    print(f"  95th percentile: {y_stats['percentile_95']:.6f}")
    print(f"  99th percentile: {y_stats['percentile_99']:.6f}")
    print(f"  99.9th percentile: {y_stats['percentile_99_9']:.6f}")
    
    print("\n" + "="*80)
    print("RECOMMENDED FALLBACK VALUES")
    print("="*80)
    print("Based on the analysis, here are some recommended fallback eta values:")
    print(f"  Conservative (99.9th percentile): eta_x = {x_stats['percentile_99_9']:.3f}, eta_y = {y_stats['percentile_99_9']:.3f}")
    print(f"  Moderate (99th percentile):       eta_x = {x_stats['percentile_99']:.3f}, eta_y = {y_stats['percentile_99']:.3f}")
    print(f"  Aggressive (95th percentile):     eta_x = {x_stats['percentile_95']:.3f}, eta_y = {y_stats['percentile_95']:.3f}")
    print(f"  Current (max error):              eta_x = {x_stats['max']:.3f}, eta_y = {y_stats['max']:.3f}")
    
    print("\n" + "="*80)
    print("SUGGESTED CODE CHANGES")
    print("="*80)
    print("Update the following constants in models/conformal_grid.py:")
    print(f"  DEFAULT_FALLBACK_ETA_X = {x_stats['percentile_99_9']:.3f}")
    print(f"  DEFAULT_FALLBACK_ETA_Y = {y_stats['percentile_99_9']:.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find maximum errors in calibration dataset")
    parser.add_argument('--calib_csv', type=str, default="assets/sf_dataset_calib.csv", 
                       help='Path to calibration CSV file (if None, collect fresh data)')
    parser.add_argument('--model_path', type=str, default='assets/best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--save_errors', action='store_true',
                       help='Save error arrays to numpy files for further analysis')
    
    args = parser.parse_args()
    
    # 查找最大误差
    stats, score_x, score_y = find_max_errors(args.calib_csv, args.model_path)
    
    # 打印分析结果
    print_error_analysis(stats)
    
    # 可选：保存误差数组
    if args.save_errors:
        np.save('calibration_errors_x.npy', score_x)
        np.save('calibration_errors_y.npy', score_y)
        print(f"\nError arrays saved to calibration_errors_x.npy and calibration_errors_y.npy")

if __name__ == "__main__":
    main()

