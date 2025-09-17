import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_data():
    """Load the CBF beta results data"""
    csv_path = Path(__file__).parent / 'cbf_beta_results.csv'
    df = pd.read_csv(csv_path)
    return df

def plot_collision_rate_vs_pedestrians(df):
    """Plot collision rate vs number of pedestrians for different beta values"""
    plt.figure(figsize=(10, 6))
    
    beta_values = sorted(df['beta'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
    
    for i, beta in enumerate(beta_values):
        beta_data = df[df['beta'] == beta]
        plt.plot(beta_data['ped_num'], beta_data['Collision_Rate'], 
                marker='o', linewidth=2, markersize=8, 
                label=f'β = {beta}', color=colors[i])
    
    plt.xlabel('Number of Pedestrians', fontsize=12)
    plt.ylabel('Collision Rate (%)', fontsize=12)
    # plt.title('CBF Collision Rate vs Number of Pedestrians\nfor Different Beta Values', fontsize=14, fontweight='bold')
    plt.legend(title='Beta Values', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['ped_num'].unique())
    
    # Add annotations for extreme values
    max_collision = df['Collision_Rate'].max()
    max_idx = df['Collision_Rate'].idxmax()
    plt.annotate(f'Max: {max_collision}%', 
                xy=(df.loc[max_idx, 'ped_num'], max_collision),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    return plt.gcf()

def plot_speed_vs_pedestrians(df):
    """Plot average speed vs number of pedestrians for different beta values"""
    plt.figure(figsize=(10, 6))
    
    beta_values = sorted(df['beta'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(beta_values)))
    
    for i, beta in enumerate(beta_values):
        beta_data = df[df['beta'] == beta]
        plt.plot(beta_data['ped_num'], beta_data['Speed_mps'], 
                marker='s', linewidth=2, markersize=8, 
                label=f'β = {beta}', color=colors[i])
    
    plt.xlabel('Number of Pedestrians', fontsize=12)
    plt.ylabel('Average Speed (m/s)', fontsize=12)
    # plt.title('CBF Average Speed vs Number of Pedestrians\nfor Different Beta Values', fontsize=14, fontweight='bold')
    plt.legend(title='Beta Values', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['ped_num'].unique())
    
    # Add trend line for overall speed decrease
    all_speeds = df.groupby('ped_num')['Speed_mps'].mean()
    z = np.polyfit(all_speeds.index, all_speeds.values, 1)
    p = np.poly1d(z)
    plt.plot(all_speeds.index, p(all_speeds.index), "k--", alpha=0.5, linewidth=1, label='Overall Trend')
    
    plt.tight_layout()
    return plt.gcf()

def plot_time_vs_pedestrians(df):
    """Plot completion time vs number of pedestrians for different beta values"""
    plt.figure(figsize=(10, 6))
    
    beta_values = sorted(df['beta'].unique())
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(beta_values)))
    
    for i, beta in enumerate(beta_values):
        beta_data = df[df['beta'] == beta]
        plt.plot(beta_data['ped_num'], beta_data['Time_ms'], 
                marker='^', linewidth=2, markersize=8, 
                label=f'β = {beta}', color=colors[i])
    
    plt.xlabel('Number of Pedestrians', fontsize=12)
    plt.ylabel('Completion Time (ms)', fontsize=12)
    # plt.title('CBF Completion Time vs Number of Pedestrians\nfor Different Beta Values', fontsize=14, fontweight='bold')
    plt.legend(title='Beta Values', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['ped_num'].unique())
    
    # Add horizontal line for average time
    avg_time = df['Time_ms'].mean()
    plt.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, 
                label=f'Average Time: {avg_time:.1f}ms')
    
    plt.tight_layout()
    return plt.gcf()

def create_heatmap_analysis(df):
    """Create a heatmap showing the relationship between beta, pedestrians, and collision rate"""
    plt.figure(figsize=(8, 6))
    
    # Pivot the data for heatmap
    heatmap_data = df.pivot(index='ped_num', columns='beta', values='Collision_Rate')
    
    # Create heatmap using matplotlib
    im = plt.imshow(heatmap_data.values, cmap='Reds', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Collision Rate (%)', fontsize=12)
    
    # Set ticks and labels
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            plt.text(j, i, f'{heatmap_data.iloc[i, j]:.1f}', 
                    ha='center', va='center', fontweight='bold')
    
    plt.title('CBF Collision Rate Heatmap\nBeta vs Number of Pedestrians', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Beta Values', fontsize=12)
    plt.ylabel('Number of Pedestrians', fontsize=12)
    
    plt.tight_layout()
    return plt.gcf()

def save_plots():
    """Generate and save all plots to the assets folder"""
    # Load data
    df = load_data()
    
    # Create assets directory if it doesn't exist
    assets_dir = Path(__file__).parent.parent / 'assets'
    assets_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating collision rate plot...")
    fig1 = plot_collision_rate_vs_pedestrians(df)
    fig1.savefig(assets_dir / 'cbf_beta_collision_rate.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating speed analysis plot...")
    fig2 = plot_speed_vs_pedestrians(df)
    fig2.savefig(assets_dir / 'cbf_beta_speed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating time analysis plot...")
    fig3 = plot_time_vs_pedestrians(df)
    fig3.savefig(assets_dir / 'cbf_beta_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating heatmap analysis...")
    fig4 = create_heatmap_analysis(df)
    fig4.savefig(assets_dir / 'cbf_beta_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"All plots saved to {assets_dir}")
    
    # Print summary statistics
    print("\n=== CBF Beta Analysis Summary ===")
    print(f"Total data points: {len(df)}")
    print(f"Beta values tested: {sorted(df['beta'].unique())}")
    print(f"Pedestrian counts: {sorted(df['ped_num'].unique())}")
    print(f"Collision rate range: {df['Collision_Rate'].min():.1f}% - {df['Collision_Rate'].max():.1f}%")
    print(f"Speed range: {df['Speed_mps'].min():.2f} - {df['Speed_mps'].max():.2f} m/s")
    print(f"Time range: {df['Time_ms'].min():.1f} - {df['Time_ms'].max():.1f} ms")
    
    # Find best performing configurations
    print("\n=== Best Performance Analysis ===")
    min_collision = df.loc[df['Collision_Rate'].idxmin()]
    max_speed = df.loc[df['Speed_mps'].idxmax()]
    min_time = df.loc[df['Time_ms'].idxmin()]
    
    print(f"Lowest collision rate: β={min_collision['beta']}, ped_num={min_collision['ped_num']} ({min_collision['Collision_Rate']}%)")
    print(f"Highest speed: β={max_speed['beta']}, ped_num={max_speed['ped_num']} ({max_speed['Speed_mps']:.2f} m/s)")
    print(f"Fastest completion: β={min_time['beta']}, ped_num={min_time['ped_num']} ({min_time['Time_ms']:.1f} ms)")

if __name__ == "__main__":
    save_plots()
