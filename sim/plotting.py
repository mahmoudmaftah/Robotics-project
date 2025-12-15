"""
Plotting utilities for simulation results.

Provides standardized plots for trajectories, phase portraits,
Lyapunov functions, and comparison charts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from typing import Optional, List, Tuple, Dict, Any
import os


def plot_trajectory(result, ax: Optional[plt.Axes] = None,
                    show_start: bool = True,
                    show_target: bool = True,
                    obstacles: Optional[List[np.ndarray]] = None,
                    goal_region: Optional[np.ndarray] = None,
                    title: Optional[str] = None,
                    color: str = 'blue',
                    label: Optional[str] = None) -> plt.Axes:
    """
    Plot 2D trajectory in state space.

    Args:
        result: SimulationResult object
        ax: Matplotlib axes (creates new if None)
        show_start: Mark initial state
        show_target: Mark target state
        obstacles: List of obstacle polygons (Nx2 arrays)
        goal_region: Goal region polygon
        title: Plot title
        color: Trajectory color
        label: Legend label

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    states = result.states

    # Plot trajectory
    ax.plot(states[:, 0], states[:, 1], '-', color=color,
            linewidth=1.5, label=label or result.metadata.get('controller', 'Trajectory'))

    # Start point
    if show_start:
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')

    # Target point
    if show_target and result.metadata.get('target') is not None:
        target = result.metadata['target']
        ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')

    # Obstacles
    if obstacles:
        for obs in obstacles:
            poly = Polygon(obs, facecolor='gray', edgecolor='black',
                          alpha=0.5, label='Obstacle')
            ax.add_patch(poly)

    # Goal region
    if goal_region is not None:
        poly = Polygon(goal_region, facecolor='green', edgecolor='darkgreen',
                      alpha=0.3, label='Goal')
        ax.add_patch(poly)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title)

    return ax


def plot_phase_portrait(result, ax: Optional[plt.Axes] = None,
                        title: Optional[str] = None) -> plt.Axes:
    """
    Plot phase portrait with velocity vectors.

    Args:
        result: SimulationResult
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    states = result.states

    # Trajectory
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5)

    # Velocity arrows (subsampled)
    skip = max(1, len(states) // 20)
    for i in range(0, len(states) - 1, skip):
        dx = states[i + 1, 0] - states[i, 0]
        dy = states[i + 1, 1] - states[i, 1]
        ax.arrow(states[i, 0], states[i, 1], dx, dy,
                head_width=0.1, head_length=0.05, fc='red', ec='red')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    return ax


def plot_lyapunov(result, ax: Optional[plt.Axes] = None,
                  log_scale: bool = True,
                  title: Optional[str] = None) -> plt.Axes:
    """
    Plot Lyapunov function evolution over time.

    Args:
        result: SimulationResult
        ax: Matplotlib axes
        log_scale: Use logarithmic y-axis
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    V = result.lyapunov_values

    if len(V) == 0 or np.all(np.isnan(V)):
        ax.text(0.5, 0.5, 'No Lyapunov data available',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    ax.plot(result.time, V, 'b-', linewidth=1.5, label='V(x)')

    if log_scale and np.all(V > 0):
        ax.set_yscale('log')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('V(x)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Lyapunov Function Evolution')

    return ax


def plot_states_and_inputs(result, state_labels: Optional[List[str]] = None,
                           input_labels: Optional[List[str]] = None,
                           title: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot state and input trajectories over time.

    Args:
        result: SimulationResult
        state_labels: Labels for state variables
        input_labels: Labels for input variables
        title: Overall figure title

    Returns:
        Figure and axes array
    """
    n_states = result.states.shape[1]
    n_inputs = result.inputs.shape[1]

    fig, axes = plt.subplots(n_states + n_inputs, 1,
                             figsize=(10, 2 * (n_states + n_inputs)),
                             sharex=True)
    axes = np.atleast_1d(axes)

    # State labels
    if state_labels is None:
        state_labels = [f'$x_{i+1}$' for i in range(n_states)]

    # Input labels
    if input_labels is None:
        input_labels = [f'$u_{i+1}$' for i in range(n_inputs)]

    # Plot states
    for i in range(n_states):
        axes[i].plot(result.time, result.states[:, i], 'b-', linewidth=1.5)
        if result.metadata.get('target') is not None:
            target = result.metadata['target']
            if i < len(target):
                axes[i].axhline(target[i], color='r', linestyle='--',
                               label='Target')
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    # Plot inputs
    for i in range(n_inputs):
        ax = axes[n_states + i]
        ax.plot(result.time, result.inputs[:, i], 'g-', linewidth=1.5)
        ax.set_ylabel(input_labels[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


def plot_comparison(results: List, labels: Optional[List[str]] = None,
                    metric: str = 'trajectory',
                    title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare multiple simulation results.

    Args:
        results: List of SimulationResult objects
        labels: Labels for each result
        metric: 'trajectory', 'lyapunov', or 'error'
        title: Plot title

    Returns:
        Figure and axes
    """
    if labels is None:
        labels = [r.metadata.get('controller', f'Controller {i}')
                  for i, r in enumerate(results)]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, ax = plt.subplots(figsize=(10, 8))

    if metric == 'trajectory':
        for result, label, color in zip(results, labels, colors):
            ax.plot(result.states[:, 0], result.states[:, 1],
                   '-', color=color, linewidth=1.5, label=label)

        # Show common start and target
        if results:
            ax.plot(results[0].states[0, 0], results[0].states[0, 1],
                   'ko', markersize=10, label='Start')
            if results[0].metadata.get('target') is not None:
                target = results[0].metadata['target']
                ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')

    elif metric == 'lyapunov':
        for result, label, color in zip(results, labels, colors):
            V = result.lyapunov_values
            if len(V) > 0:
                ax.semilogy(result.time, V, '-', color=color,
                           linewidth=1.5, label=label)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('V(x)')

    elif metric == 'error':
        for result, label, color in zip(results, labels, colors):
            errors = result.error_norms
            if len(errors) > 0:
                ax.semilogy(result.time, errors, '-', color=color,
                           linewidth=1.5, label=label)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error norm')

    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax


def save_figure(fig: plt.Figure, filename: str,
                output_dir: str = 'report/figures',
                formats: List[str] = ['png', 'pdf']) -> None:
    """
    Save figure to multiple formats.

    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of file formats
    """
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        path = os.path.join(output_dir, f'{filename}.{fmt}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
