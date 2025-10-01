#!/usr/bin/env python3
"""
Comparison script for RL vs MOBO approaches to PPA optimization.

Usage:
    python compare.py --rl-results runs/rl_20250930_120000 --mobo-results runs/mobo_20250930_130000
    python compare.py --scan-latest  # Automatically find latest RL and MOBO results
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from shared.utils.pareto import pareto_from_dicts, convert_rl_to_mobo_format, pareto_from_tensors
from shared.utils.plotting import plot_pareto_comparison, plot_convergence_comparison


def load_rl_results(rl_dir: str) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """Load RL results from run directory."""
    rl_path = Path(rl_dir)

    # Load candidates
    candidates_path = rl_path / "candidates.csv"
    progress_path = rl_path / "progress.csv"

    candidates = []
    progress = None

    if candidates_path.exists():
        df = pd.read_csv(candidates_path)
        candidates = df.to_dict('records')

    if progress_path.exists():
        progress = pd.read_csv(progress_path).to_dict('records')

    return candidates, progress


def load_mobo_results(mobo_dir: str) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """Load MOBO results from run directory."""
    mobo_path = Path(mobo_dir)

    candidates_path = mobo_path / "candidates.csv"
    pareto_path = mobo_path / "pareto.csv"
    progress_path = mobo_path / "progress.csv"

    Y_all = np.empty((0, 2))
    pareto_Y = np.empty((0, 2))
    progress = None

    if candidates_path.exists():
        df = pd.read_csv(candidates_path)
        Y_all = df[['obj_0', 'obj_1']].values

    if pareto_path.exists():
        df = pd.read_csv(pareto_path)
        pareto_Y = df[['obj_0', 'obj_1']].values

    if progress_path.exists():
        progress = pd.read_csv(progress_path).to_dict('records')

    return Y_all, pareto_Y, progress


def find_latest_results(runs_dir: str = "runs") -> Tuple[Optional[str], Optional[str]]:
    """Find the latest RL and MOBO result directories."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None, None

    rl_dirs = sorted([d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith("rl_")])
    mobo_dirs = sorted([d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith("mobo_")])

    latest_rl = str(rl_dirs[-1]) if rl_dirs else None
    latest_mobo = str(mobo_dirs[-1]) if mobo_dirs else None

    return latest_rl, latest_mobo


def compute_metrics(rl_candidates: List[Dict], mobo_Y: np.ndarray, mobo_pareto_Y: np.ndarray) -> Dict:
    """Compute comparison metrics."""
    metrics = {}

    # RL metrics
    if rl_candidates:
        rl_pareto = pareto_from_dicts(rl_candidates)
        metrics['rl_total_evaluations'] = len(rl_candidates)
        metrics['rl_pareto_size'] = len(rl_pareto)

        if rl_pareto:
            rl_power = [p['power'] for p in rl_pareto]
            rl_delay = [p['delay'] for p in rl_pareto]
            metrics['rl_pareto_power_range'] = (min(rl_power), max(rl_power))
            metrics['rl_pareto_delay_range'] = (min(rl_delay), max(rl_delay))

    # MOBO metrics
    metrics['mobo_total_evaluations'] = len(mobo_Y)
    metrics['mobo_pareto_size'] = len(mobo_pareto_Y)

    if len(mobo_pareto_Y) > 0:
        metrics['mobo_pareto_power_range'] = (mobo_pareto_Y[:, 0].min(), mobo_pareto_Y[:, 0].max())
        metrics['mobo_pareto_delay_range'] = (mobo_pareto_Y[:, 1].min(), mobo_pareto_Y[:, 1].max())

    # Hypervolume comparison (simplified - assuming same reference point)
    if rl_candidates and len(mobo_pareto_Y) > 0:
        # Convert RL to tensor format for comparison
        rl_Y = convert_rl_to_mobo_format(rl_candidates)
        rl_pareto_idx, _ = pareto_from_tensors(rl_Y)
        rl_pareto_Y = rl_Y[rl_pareto_idx]

        # Simple hypervolume approximation (area under curve for 2D)
        def simple_hv_2d(pareto_points, ref_point):
            if len(pareto_points) == 0:
                return 0.0
            # Sort by first objective
            sorted_points = pareto_points[np.argsort(pareto_points[:, 0])]
            hv = 0.0
            for i, (x, y) in enumerate(sorted_points):
                if i == 0:
                    width = ref_point[0] - x
                else:
                    width = sorted_points[i-1, 0] - x
                height = ref_point[1] - y
                hv += width * height
            return hv

        # Use worst observed + 10% as reference point
        all_points = np.vstack([rl_Y.numpy(), mobo_Y])
        ref_point = all_points.max(axis=0) * 1.1

        metrics['rl_hypervolume'] = simple_hv_2d(rl_pareto_Y.numpy(), ref_point)
        metrics['mobo_hypervolume'] = simple_hv_2d(mobo_pareto_Y, ref_point)
        metrics['hv_improvement'] = (metrics['mobo_hypervolume'] - metrics['rl_hypervolume']) / max(metrics['rl_hypervolume'], 1e-12)

    return metrics


def generate_comparison_report(metrics: Dict, rl_dir: str, mobo_dir: str) -> str:
    """Generate a formatted comparison report as a string."""
    report = []
    report.append("=" * 80)
    report.append("RL vs MOBO PPA Optimization Comparison")
    report.append("=" * 80)
    report.append(f"RL Results:   {rl_dir}")
    report.append(f"MOBO Results: {mobo_dir}")
    report.append("-" * 80)

    report.append(f"Total Evaluations:")
    report.append(f"  RL:   {metrics.get('rl_total_evaluations', 'N/A')}")
    report.append(f"  MOBO: {metrics.get('mobo_total_evaluations', 'N/A')}")

    report.append(f"\nPareto Frontier Size:")
    report.append(f"  RL:   {metrics.get('rl_pareto_size', 'N/A')}")
    report.append(f"  MOBO: {metrics.get('mobo_pareto_size', 'N/A')}")

    if 'rl_hypervolume' in metrics and 'mobo_hypervolume' in metrics:
        report.append(f"\nHypervolume:")
        report.append(f"  RL:         {metrics['rl_hypervolume']:.6f}")
        report.append(f"  MOBO:       {metrics['mobo_hypervolume']:.6f}")
        report.append(f"  Improvement: {metrics['hv_improvement']:.2%}")

    if 'rl_pareto_power_range' in metrics:
        rl_pr, rl_dr = metrics['rl_pareto_power_range'], metrics['rl_pareto_delay_range']
        report.append(f"\nRL Pareto Ranges:")
        report.append(f"  Power: [{rl_pr[0]:.4f}, {rl_pr[1]:.4f}]")
        report.append(f"  Delay: [{rl_dr[0]:.4f}, {rl_dr[1]:.4f}]")

    if 'mobo_pareto_power_range' in metrics:
        mobo_pr, mobo_dr = metrics['mobo_pareto_power_range'], metrics['mobo_pareto_delay_range']
        report.append(f"\nMOBO Pareto Ranges:")
        report.append(f"  Power: [{mobo_pr[0]:.4f}, {mobo_pr[1]:.4f}]")
        report.append(f"  Delay: [{mobo_dr[0]:.4f}, {mobo_dr[1]:.4f}]")

    report.append("=" * 80)
    return "\n".join(report)


def save_comparison_report(report: str, output_file: str):
    """Save comparison report to file."""
    with open(output_file, 'w') as f:
        f.write(report)


def print_comparison_report(metrics: Dict, rl_dir: str, mobo_dir: str):
    """Print a formatted comparison report to console."""
    report = generate_comparison_report(metrics, rl_dir, mobo_dir)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Compare RL and MOBO PPA optimization results")
    parser.add_argument("--rl-results", type=str, help="Path to RL results directory")
    parser.add_argument("--mobo-results", type=str, help="Path to MOBO results directory")
    parser.add_argument("--scan-latest", action="store_true", help="Automatically find latest results")
    parser.add_argument("--output-dir", type=str, default="comparison_plots", help="Output directory for plots")

    args = parser.parse_args()

    # Determine result directories
    if args.scan_latest:
        rl_dir, mobo_dir = find_latest_results()
        if not rl_dir or not mobo_dir:
            print("Could not find latest RL and/or MOBO results. Please specify manually.")
            return
    else:
        rl_dir, mobo_dir = args.rl_results, args.mobo_results
        if not rl_dir or not mobo_dir:
            print("Please specify both --rl-results and --mobo-results or use --scan-latest")
            return

    print(f"Comparing RL results from: {rl_dir}")
    print(f"         MOBO results from: {mobo_dir}")

    # Load results
    rl_candidates, rl_progress = load_rl_results(rl_dir)
    mobo_Y, mobo_pareto_Y, mobo_progress = load_mobo_results(mobo_dir)

    # Compute metrics
    metrics = compute_metrics(rl_candidates, mobo_Y, mobo_pareto_Y)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate and save report
    report = generate_comparison_report(metrics, rl_dir, mobo_dir)
    report_file = output_dir / "comparison_report.txt"
    save_comparison_report(report, str(report_file))

    # Also print to console
    print(report)
    print(f"\nComparison report saved to: {report_file}")

    # Generate plots
    print(f"\nGenerating comparison plots in {output_dir}/...")

    # Pareto frontier comparison
    plot_pareto_comparison(
        rl_candidates, mobo_Y, mobo_pareto_Y,
        output_dir / "pareto_comparison.png",
        "RL vs MOBO Pareto Frontier Comparison"
    )

    # Convergence comparison
    plot_convergence_comparison(
        rl_progress, mobo_progress,
        output_dir / "convergence_comparison.png",
        "RL vs MOBO Convergence Comparison"
    )

    print("Comparison complete!")


if __name__ == "__main__":
    main()