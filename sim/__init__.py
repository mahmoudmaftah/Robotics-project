"""
Simulation engine and plotting utilities.

Provides controller-agnostic simulation with disturbance injection,
visualization, and animation export.
"""

from .simulator import Simulator, SimulationResult
from .plotting import plot_trajectory, plot_phase_portrait, plot_lyapunov
from .animation import (
    AnimationExporter,
    export_simulation,
    create_comparison_animation
)

__all__ = [
    'Simulator',
    'SimulationResult',
    'plot_trajectory',
    'plot_phase_portrait',
    'plot_lyapunov',
    'AnimationExporter',
    'export_simulation',
    'create_comparison_animation'
]
