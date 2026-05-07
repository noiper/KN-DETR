import csv
import os
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not found.")
    exit(1)

def plot_analysis(csv_path: str):
    csv_p = Path(csv_path)
    if not csv_p.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Automatic output naming
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    output_image = plots_dir / (csv_p.stem + ".png")

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader: rows.append(r)

    scores, errors, gaps = [], [], []
    for r in rows:
        try:
            local_v = float(r['max_abs_diff_s5_prev'])
            drift_v = float(r['l1_s5_anchor'])
            map_err = float(r['map_gap'])
            gap = int(r['frame_gap'])
            if np.isfinite(local_v) and np.isfinite(drift_v) and np.isfinite(map_err):
                # Using our standard detector score for X-axis
                scores.append(2.0 * local_v + drift_v)
                errors.append(map_err)
                gaps.append(gap)
        except: continue

    if not scores:
        print("No valid data points found to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot A: Score vs Error
    scatter = ax1.scatter(scores, errors, c=gaps, cmap='viridis', alpha=0.6)
    fig.colorbar(scatter, ax=ax1, label='Frame Gap')
    ax1.set_xlabel('Hard-Frame Score (2*Local + Drift)')
    ax1.set_ylabel('mAP Drop (Oracle - NonKey)')
    ax1.set_title(f'Routing Sensitivity: {csv_p.stem}')
    ax1.grid(True, alpha=0.3)

    # Plot B: Gap vs Error (Boxplot)
    max_gap = max(gaps) if gaps else 1
    ax2.boxplot([np.array(errors)[np.array(gaps) == i] for i in range(1, max_gap + 1)], labels=range(1, max_gap + 1))
    ax2.set_xlabel('Frame Gap')
    ax2.set_ylabel('mAP Drop')
    ax2.set_title('Degradation over Temporal Distance')

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Success: Plot saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to the analysis CSV file")
    args = parser.parse_args()
    plot_analysis(args.csv_path)

