"""Shared plotting utilities for RL and MOBO approaches."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional


def plot_pareto_comparison(rl_results: List[Dict], mobo_Y: np.ndarray, mobo_pareto_Y: np.ndarray,
                          output_path: str, title: str = "RL vs MOBO Pareto Comparison"):
    """
    Create side-by-side comparison plot of RL and MOBO Pareto frontiers.

    Args:
        rl_results: List of dicts with RL results
        mobo_Y: All MOBO candidates (n_points, 2)
        mobo_pareto_Y: MOBO Pareto frontier (n_pareto, 2)
        output_path: Where to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # RL plot
    if rl_results:
        power_vals = [p['power'] for p in rl_results]
        delay_vals = [p['delay'] for p in rl_results]
        ax1.scatter(power_vals, delay_vals, alpha=0.6, label='RL candidates')

        # Extract RL Pareto frontier
        from shared.utils.pareto import pareto_from_dicts
        rl_pareto = pareto_from_dicts(rl_results)
        if rl_pareto:
            rl_p_power = [p['power'] for p in rl_pareto]
            rl_p_delay = [p['delay'] for p in rl_pareto]
            sorted_indices = np.argsort(rl_p_power)
            ax1.plot([rl_p_power[i] for i in sorted_indices],
                    [rl_p_delay[i] for i in sorted_indices],
                    'r-o', label='RL Pareto', linewidth=2)

    ax1.set_xlabel('Power (min)')
    ax1.set_ylabel('Delay (min)')
    ax1.set_title('RL Approach')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MOBO plot
    ax2.scatter(mobo_Y[:, 0], mobo_Y[:, 1], alpha=0.6, label='MOBO candidates')
    if len(mobo_pareto_Y) > 0:
        sorted_indices = np.argsort(mobo_pareto_Y[:, 0])
        ax2.plot(mobo_pareto_Y[sorted_indices, 0], mobo_pareto_Y[sorted_indices, 1],
                'r-o', label='MOBO Pareto', linewidth=2)

    ax2.set_xlabel('Power (min)')
    ax2.set_ylabel('Delay (min)')
    ax2.set_title('MOBO Approach')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_comparison(rl_progress: Optional[List[Dict]],
                               mobo_progress: Optional[List[Dict]],
                               output_path: str,
                               title: str = "Convergence Comparison"):
    """
    Plot convergence curves comparing RL and MOBO approaches.

    Args:
        rl_progress: RL training progress data
        mobo_progress: MOBO iteration progress data
        output_path: Where to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # RL convergence (reward over time)
    if rl_progress:
        timesteps = [p.get('time/total_timesteps', i) for i, p in enumerate(rl_progress)]
        rewards = [p.get('rollout/ep_rew_mean', 0) for p in rl_progress]
        ax1.plot(timesteps, rewards, 'b-', label='Mean Reward')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Reward')
        ax1.set_title('RL Training Progress')
        ax1.grid(True, alpha=0.3)

    # MOBO convergence (hypervolume over iterations)
    if mobo_progress:
        iterations = [p.get('iter', i) for i, p in enumerate(mobo_progress)]
        hypervolumes = [p.get('hv', 0) for p in mobo_progress]
        ax2.plot(iterations, hypervolumes, 'g-o', label='Hypervolume')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Hypervolume')
        ax2.set_title('MOBO Optimization Progress')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_single_pareto(candidates: np.ndarray, pareto_points: np.ndarray,
                      output_path: str, title: str = "Pareto Frontier",
                      obj_labels: Tuple[str, str] = ("Objective 1", "Objective 2")):
    """
    Plot a single Pareto frontier with candidates.

    Args:
        candidates: All candidate points (n_points, 2)
        pareto_points: Pareto optimal points (n_pareto, 2)
        output_path: Where to save the plot
        title: Plot title
        obj_labels: Labels for the two objectives
    """
    plt.figure(figsize=(8, 6))

    # Plot all candidates
    plt.scatter(candidates[:, 0], candidates[:, 1], alpha=0.6, color='lightblue',
               s=30, label='Candidates')

    # Plot Pareto frontier
    if len(pareto_points) > 0:
        sorted_indices = np.argsort(pareto_points[:, 0])
        plt.plot(pareto_points[sorted_indices, 0], pareto_points[sorted_indices, 1],
                'r-o', label='Pareto Frontier', linewidth=2, markersize=6)

    plt.xlabel(obj_labels[0])
    plt.ylabel(obj_labels[1])
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_parallel_coordinates(data: np.ndarray, pareto_mask: np.ndarray,
                             output_path: str, var_names: List[str],
                             title: str = "Parallel Coordinates"):
    """
    Create parallel coordinates plot showing design variables.

    Args:
        data: Design variables (n_points, n_vars)
        pareto_mask: Boolean mask indicating Pareto optimal points
        output_path: Where to save the plot
        var_names: Names of design variables
        title: Plot title
    """
    n_vars = data.shape[1]
    if n_vars < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize data to [0, 1] for plotting
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-12)

    # Plot non-Pareto points
    if (~pareto_mask).any():
        for i in range(data_norm.shape[0]):
            if not pareto_mask[i]:
                ax.plot(range(n_vars), data_norm[i], 'b-', alpha=0.1)

    # Plot Pareto points
    if pareto_mask.any():
        for i in range(data_norm.shape[0]):
            if pareto_mask[i]:
                ax.plot(range(n_vars), data_norm[i], 'r-', alpha=0.7, linewidth=2)

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(var_names)
    ax.set_ylabel('Normalized Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', alpha=0.3, label='Non-Pareto'),
        Line2D([0], [0], color='red', alpha=0.7, linewidth=2, label='Pareto Optimal')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()