"""Unified Pareto frontier computation for both RL and MOBO approaches."""

import numpy as np
import torch
from typing import List, Dict, Union, Tuple


def pareto_from_dicts(points: List[Dict], objectives: List[str] = None) -> List[Dict]:
    """
    Compute Pareto frontier from list of dictionaries (RL format).

    Args:
        points: List of dicts with objective values to MINIMIZE
        objectives: List of objective keys to consider (default: ['power', 'delay'])

    Returns:
        Non-dominated subset sorted by first objective
    """
    if not points:
        return []

    if objectives is None:
        objectives = ['power', 'delay']

    # Sort by first objective, then by second
    sorted_points = sorted(points, key=lambda d: tuple(d[obj] for obj in objectives))

    # Skyline algorithm for 2D case (can extend for higher dimensions)
    if len(objectives) == 2:
        front = []
        best_second = float('inf')

        for point in sorted_points:
            current_second = point[objectives[1]]
            if current_second < best_second:
                front.append(point)
                best_second = current_second

        return front
    else:
        # General n-dimensional non-dominated sorting
        return _general_pareto_filter(sorted_points, objectives)


def pareto_from_tensors(Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Pareto frontier from tensor of objectives (MOBO format).

    Args:
        Y: Tensor of shape (n_points, n_objectives) with values to MINIMIZE

    Returns:
        (pareto_indices, pareto_mask): Indices and boolean mask of non-dominated points
    """
    try:
        from botorch.utils.multi_objective import is_non_dominated
        mask = is_non_dominated(Y)
        indices = torch.arange(Y.shape[0])[mask]
        return indices, mask
    except ImportError:
        # Fallback implementation if BoTorch not available
        return _manual_pareto_tensors(Y)


def _manual_pareto_tensors(Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manual implementation of Pareto filtering for tensors."""
    n_points, n_obj = Y.shape
    is_dominated = torch.zeros(n_points, dtype=torch.bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Point i is dominated by point j if j is better in all objectives
                # and strictly better in at least one
                better_all = (Y[j] <= Y[i]).all()
                better_any = (Y[j] < Y[i]).any()
                if better_all and better_any:
                    is_dominated[i] = True
                    break

    mask = ~is_dominated
    indices = torch.arange(n_points)[mask]
    return indices, mask


def _general_pareto_filter(points: List[Dict], objectives: List[str]) -> List[Dict]:
    """General n-dimensional Pareto filtering."""
    n_points = len(points)
    is_dominated = [False] * n_points

    for i in range(n_points):
        for j in range(n_points):
            if i != j and not is_dominated[i]:
                # Check if point j dominates point i
                better_all = all(points[j][obj] <= points[i][obj] for obj in objectives)
                better_any = any(points[j][obj] < points[i][obj] for obj in objectives)

                if better_all and better_any:
                    is_dominated[i] = True
                    break

    return [points[i] for i in range(n_points) if not is_dominated[i]]


def convert_rl_to_mobo_format(rl_results: List[Dict], objectives: List[str] = None) -> torch.Tensor:
    """Convert RL results format to MOBO tensor format."""
    if objectives is None:
        objectives = ['power', 'delay']

    if not rl_results:
        return torch.empty(0, len(objectives))

    Y = torch.tensor([[point[obj] for obj in objectives] for point in rl_results])
    return Y


def convert_mobo_to_rl_format(X: torch.Tensor, Y: torch.Tensor,
                              var_names: List[str] = None,
                              obj_names: List[str] = None) -> List[Dict]:
    """Convert MOBO tensor format to RL results format."""
    if var_names is None:
        var_names = [f'x_{i}' for i in range(X.shape[1])]
    if obj_names is None:
        obj_names = [f'obj_{i}' for i in range(Y.shape[1])]

    results = []
    for i in range(X.shape[0]):
        point = {}
        # Add design variables
        for j, name in enumerate(var_names):
            point[name] = X[i, j].item()
        # Add objectives
        for j, name in enumerate(obj_names):
            point[name] = Y[i, j].item()
        results.append(point)

    return results