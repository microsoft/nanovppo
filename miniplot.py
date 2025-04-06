import argparse
import os
import json
import torch
import matplotlib
matplotlib.use('Agg')  # Force TkAgg backend for X server compatibility
import matplotlib.pyplot as plt


def plot_metrics(groups, metrics=['avg_reward'], xlim=None, output_file='plot.png'):
    fig, ax = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        ax = [ax]

    for name, file_path in groups.items():
        with open(file_path, 'r') as f:
            data = json.load(f)

        for i, metric in enumerate(metrics):
            metric_data = [d[metric] for d in data if metric in d]
            ax[i].plot(
                [d['epoch'] for d in data if metric in d], 
                metric_data, 
                label=name if i == 0 else None, 
                linewidth=2
            )
            ax[i].set_xlabel('epoch')
            ax[i].set_ylabel(metric)
            ax[i].grid(color='white', linestyle='-', linewidth=0.5)
            ax[i].patch.set_facecolor('lightgray')
            ax[i].patch.set_alpha(0.5)
            for spine in ax[i].spines.values():
                spine.set_linewidth(2)
            ax[i].tick_params(width=2)
            if xlim:
                ax[i].set_xlim(*xlim)

    fig.legend(loc='upper right', bbox_to_anchor=(0.5, 1.1), ncol=len(metrics))
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save to file
    plt.close(fig)  # Close to free memory


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from JSON files')
    parser.add_argument('paths', nargs='+', 
                       help='Paths to results directories containing training_stats.json')
    parser.add_argument('--metrics', nargs='+', default=['avg_reward'],
                       help='Metrics to plot (default: avg_reward)')
    parser.add_argument('--xlim', nargs=2, type=float, default=None,
                       help='X-axis limits (min max)')
    parser.add_argument('--output', type=str, default='plot.png',
                       help='Output file name (default: plot.png)')

    args = parser.parse_args()

    groups = {}
    for path in args.paths:
        name = os.path.basename(os.path.normpath(path))
        file_path = os.path.join(path, "training_stats.json")
        if not os.path.exists(file_path):
            print(f"Warning: training_stats.json not found in {path}")
            continue
        groups[name] = file_path

    if not groups:
        print("Error: No valid training_stats.json files found")
        return

    plot_metrics(groups, metrics=args.metrics, xlim=args.xlim, output_file=args.output)


if __name__ == '__main__':
    main()
