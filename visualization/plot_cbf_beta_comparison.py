import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
# RESULTS_DIR = PROJECT_ROOT / "assets"
DATA_CSV = Path(__file__).resolve().parent / "cbf_beta_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_CSV)
    # Normalize columns
    df["ped_num"] = df["ped_num"].astype(int)
    # Ensure numeric
    df["Collision_Rate"] = pd.to_numeric(df["Collision_Rate"], errors="coerce")
    df["Speed_mps"] = pd.to_numeric(df["Speed_mps"], errors="coerce")
    df["Time_ms"] = pd.to_numeric(df["Time_ms"], errors="coerce")
    # Keep only rows with meaningful data
    df = df.dropna(subset=["Collision_Rate"], how="any")
    return df


def style_matplotlib():
    # Use pure Matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)
    # Enlarge global font sizes for all texts
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12


def plot_collision_figure(df: pd.DataFrame):
    # Prepare beta as string while keeping original for labels
    df = df.copy()
    df["beta_str"] = df["beta"].astype(str).str.strip()

    # Aggregate duplicates per (beta, ped_num) by mean
    agg = (
        df.groupby(["beta_str", "ped_num"], as_index=False)["Collision_Rate"]
        .mean()
    )

    # Order: Random first, then numeric betas ascending
    def beta_sort_key(b):
        if b == "Random":
            return (-1, -1.0)
        try:
            return (0, float(b))
        except ValueError:
            return (1, float("inf"))

    unique_betas = sorted(agg["beta_str"].unique().tolist(), key=beta_sort_key)

    # Figure
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # Fixed color mapping per beta - using more distinct colors
    def color_for_beta(b: str):
        if b in ["0.25", "0.25"]:
            return "#2ca02c"  # green
        if b in ["0.5", "0.5"]:
            return "#ff7f0e"  # orange
        if b in ["1", "1.0"]:
            return "#9467bd"  # purple
        if b in ["10", "10.0"]:
            return "#d62728"  # red
        if b == "Random":
            return "#8c564b"  # brown
        # fallback to default cycle
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        return cycle[0] if cycle else "#000000"

    # Plot order: Random first, then ascending beta values
    for b in unique_betas:
        grp = agg[agg["beta_str"] == b].sort_values("ped_num")
        color = color_for_beta(b)
        # Label naming
        if b == "Random":
            label = "Random"
        else:
            label = f"CBF (β={b})"
        ax.plot(grp["ped_num"], grp["Collision_Rate"], "o-", label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Number of pedestrians")
    ax.set_ylabel("Collision rate (%)")
    # ax.set_title("CBF Collision Rate vs Pedestrians\nfor Different Beta Values")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["ped_num"].unique())
    # Use rcParams tick sizes
    fig.tight_layout()
    return fig


def plot_speed_figure(df: pd.DataFrame):
    # Prepare beta as string while keeping original for labels
    df = df.copy()
    df["beta_str"] = df["beta"].astype(str).str.strip()

    # Aggregate duplicates per (beta, ped_num) by mean for speed
    agg = (
        df.groupby(["beta_str", "ped_num"], as_index=False)["Speed_mps"]
        .mean()
    )

    # Order: Random first, then numeric betas ascending
    def beta_sort_key(b):
        if b == "Random":
            return (-1, -1.0)
        try:
            return (0, float(b))
        except ValueError:
            return (1, float("inf"))

    unique_betas = sorted(agg["beta_str"].unique().tolist(), key=beta_sort_key)
    # Exclude Random from speed plot
    unique_betas = [b for b in unique_betas if b != "Random"]

    # Figure
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # Fixed color mapping per beta (same as collision figure)
    beta_color = {
        "0.25": "#2ca02c",  # green
        "0.5": "#ff7f0e",   # orange
        "1": "#9467bd",     # purple
        "1.0": "#9467bd",   # purple
        "10": "#d62728",    # red
        "10.0": "#d62728",  # red
        "Random": "#8c564b", # brown
    }

    for b in unique_betas:
        grp = agg[agg["beta_str"] == b].sort_values("ped_num")
        color = beta_color.get(b, plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#000000"])[0])
        label = f"CBF (β={b})"
        ax.plot(grp["ped_num"], grp["Speed_mps"], "s--", label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Number of pedestrians")
    ax.set_ylabel("Average speed (m/s)")
    # ax.set_title("CBF Average Speed vs Pedestrians\nfor Different Beta Values")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["ped_num"].unique())
    # Use rcParams tick sizes
    fig.tight_layout()
    return fig


def plot_time_figure(df: pd.DataFrame):
    # Prepare beta as string while keeping original for labels
    df = df.copy()
    df["beta_str"] = df["beta"].astype(str).str.strip()

    # Aggregate duplicates per (beta, ped_num) by mean for time
    agg = (
        df.groupby(["beta_str", "ped_num"], as_index=False)["Time_ms"]
        .mean()
    )

    # Order: Random first, then numeric betas ascending
    def beta_sort_key(b):
        if b == "Random":
            return (-1, -1.0)
        try:
            return (0, float(b))
        except ValueError:
            return (1, float("inf"))

    unique_betas = sorted(agg["beta_str"].unique().tolist(), key=beta_sort_key)
    # Exclude Random from time plot
    unique_betas = [b for b in unique_betas if b != "Random"]

    # Figure
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # Fixed color mapping per beta (same as other figures)
    beta_color = {
        "0.25": "#2ca02c",  # green
        "0.5": "#ff7f0e",   # orange
        "1": "#9467bd",     # purple
        "1.0": "#9467bd",   # purple
        "10": "#d62728",    # red
        "10.0": "#d62728",  # red
        "Random": "#8c564b", # brown
    }

    for b in unique_betas:
        grp = agg[agg["beta_str"] == b].sort_values("ped_num")
        color = beta_color.get(b, plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#000000"])[0])
        label = f"CBF (β={b})"
        ax.plot(grp["ped_num"], grp["Time_ms"], "^-.", label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Number of pedestrians")
    ax.set_ylabel("Calculation time (ms)")
    # ax.set_title("CBF Calculation Time vs Pedestrians\nfor Different Beta Values")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["ped_num"].unique())
    # Use rcParams tick sizes
    fig.tight_layout()
    return fig


def create_combined_figure(df: pd.DataFrame):
    """Create a combined figure with all three subplots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Prepare data
    df = df.copy()
    df["beta_str"] = df["beta"].astype(str).str.strip()
    
    # Color mapping
    beta_color = {
        "0.25": "#2ca02c",  # green
        "0.5": "#ff7f0e",   # orange
        "1": "#9467bd",     # purple
        "1.0": "#9467bd",   # purple
        "10": "#d62728",    # red
        "10.0": "#d62728",  # red
    }
    
    # Beta order
    def beta_sort_key(b):
        try:
            return float(b)
        except ValueError:
            return float("inf")
    
    unique_betas = sorted(df["beta_str"].unique().tolist(), key=beta_sort_key)
    
    # Plot 1: Collision Rate
    ax1 = axes[0]
    for b in unique_betas:
        grp = df[df["beta_str"] == b].sort_values("ped_num")
        color = beta_color.get(b, "#000000")
        label = f"CBF (β={b})"
        ax1.plot(grp["ped_num"], grp["Collision_Rate"], "o-", label=label, color=color, linewidth=2, markersize=5)
    
    ax1.set_xlabel("Number of pedestrians")
    ax1.set_ylabel("Collision rate (%)")
    ax1.set_title("(a) Collision Rate")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df["ped_num"].unique())
    
    # Plot 2: Speed
    ax2 = axes[1]
    for b in unique_betas:
        grp = df[df["beta_str"] == b].sort_values("ped_num")
        color = beta_color.get(b, "#000000")
        label = f"CBF (β={b})"
        ax2.plot(grp["ped_num"], grp["Speed_mps"], "s--", label=label, color=color, linewidth=2, markersize=5)
    
    ax2.set_xlabel("Number of pedestrians")
    ax2.set_ylabel("Average speed (m/s)")
    ax2.set_title("(b) Average Speed")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df["ped_num"].unique())
    
    # Plot 3: Time
    ax3 = axes[2]
    for b in unique_betas:
        grp = df[df["beta_str"] == b].sort_values("ped_num")
        color = beta_color.get(b, "#000000")
        label = f"CBF (β={b})"
        ax3.plot(grp["ped_num"], grp["Time_ms"], "^-.", label=label, color=color, linewidth=2, markersize=5)
    
    ax3.set_xlabel("Number of pedestrians")
    ax3.set_ylabel("Calculation time (ms)")
    ax3.set_title("(c) Calculation Time")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df["ped_num"].unique())
    
    # Add legend to the middle plot
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def main():
    style_matplotlib()
    df = load_data()
    
    # Generate individual figures
    fig_collision = plot_collision_figure(df)
    fig_speed = plot_speed_figure(df)
    fig_time = plot_time_figure(df)
    fig_combined = create_combined_figure(df)
    
    # Save individual figures to assets
    png_collision = RESULTS_DIR / "cbf_beta_collision_comparison.png"
    pdf_collision = RESULTS_DIR / "cbf_beta_collision_comparison.pdf"
    fig_collision.savefig(png_collision, bbox_inches="tight", dpi=300)
    fig_collision.savefig(pdf_collision, bbox_inches="tight")
    print(f"Saved: {png_collision}")
    print(f"Saved: {pdf_collision}")
    
    png_speed = RESULTS_DIR / "cbf_beta_speed_comparison.png"
    pdf_speed = RESULTS_DIR / "cbf_beta_speed_comparison.pdf"
    fig_speed.savefig(png_speed, bbox_inches="tight", dpi=300)
    fig_speed.savefig(pdf_speed, bbox_inches="tight")
    print(f"Saved: {png_speed}")
    print(f"Saved: {pdf_speed}")
    
    png_time = RESULTS_DIR / "cbf_beta_time_comparison.png"
    pdf_time = RESULTS_DIR / "cbf_beta_time_comparison.pdf"
    fig_time.savefig(png_time, bbox_inches="tight", dpi=300)
    fig_time.savefig(pdf_time, bbox_inches="tight")
    print(f"Saved: {png_time}")
    print(f"Saved: {pdf_time}")
    
    # Save combined figure
    png_combined = RESULTS_DIR / "cbf_beta_combined_comparison.png"
    pdf_combined = RESULTS_DIR / "cbf_beta_combined_comparison.pdf"
    fig_combined.savefig(png_combined, bbox_inches="tight", dpi=300)
    fig_combined.savefig(pdf_combined, bbox_inches="tight")
    print(f"Saved: {png_combined}")
    print(f"Saved: {pdf_combined}")
    
    # Print summary
    print("\n=== CBF Beta Analysis Summary ===")
    print(f"Total data points: {len(df)}")
    print(f"Beta values tested: {sorted(df['beta'].unique())}")
    print(f"Pedestrian counts: {sorted(df['ped_num'].unique())}")
    print(f"Collision rate range: {df['Collision_Rate'].min():.1f}% - {df['Collision_Rate'].max():.1f}%")
    print(f"Speed range: {df['Speed_mps'].min():.2f} - {df['Speed_mps'].max():.2f} m/s")
    print(f"Time range: {df['Time_ms'].min():.1f} - {df['Time_ms'].max():.1f} ms")


if __name__ == "__main__":
    main()
