#!/usr/bin/env python3
"""
Smoke test for Model 2 (Unicycle).

Tests all unicycle controllers:
(a) Feedback Linearization
(b) Polar Coordinate Controller
(c) Sliding Mode Control
(d) LQR (linearized)

Based on discrete-time unicycle from Symbolic_control_lecture-7.pdf:
    x1(t+1) = x1(t) + tau * (v * cos(theta) + w1)
    x2(t+1) = x2(t) + tau * (v * sin(theta) + w2)
    theta(t+1) = theta(t) + tau * (omega + w3)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unicycle import UnicycleModel
from controllers.unicycle import (
    FeedbackLinearizationController,
    PolarCoordinateController,
    SlidingModeController,
    UnicycleLQRController
)
from sim.simulator import Simulator
from sim.plotting import save_figure


def ensure_dirs():
    """Create output directories."""
    os.makedirs('report/figures', exist_ok=True)


def run_controller_test(model, controller, x0, target, T=20.0):
    """Run a single controller test."""
    controller.set_model(model)
    controller.set_target(target)

    sim = Simulator(model, controller, disturbance_mode='random', seed=42)
    result = sim.run(x0, T)

    return result


def main():
    """Run unicycle smoke test."""
    print("=" * 60)
    print("SMOKE TEST: Model 2 (Unicycle)")
    print("=" * 60)

    ensure_dirs()

    # Model setup
    model = UnicycleModel(tau=0.1)
    print(f"\nModel: {model}")
    print(f"  Sampling period: tau = {model.tau}")
    print(f"  State bounds: {model.x_bounds.tolist()}")
    print(f"  Input bounds: {model.u_bounds.tolist()}")

    # Test configuration
    x0 = np.array([2.0, 2.0, 0.0])  # Start at (2,2) facing right
    target = np.array([8.0, 8.0, np.pi/2])  # Go to (8,8) facing up
    T = 25.0

    print(f"\nTest configuration:")
    print(f"  Initial state: x0 = {x0}")
    print(f"  Target state: x* = {target}")
    print(f"  Simulation time: T = {T} s")

    # Controllers to test
    controllers = [
        ("Feedback Linearization", FeedbackLinearizationController(d=0.3, kp=1.0)),
        ("Polar Coordinate", PolarCoordinateController(k_rho=1.0, k_alpha=3.0, k_beta=-0.5)),
        ("Sliding Mode", SlidingModeController(k_pos=2.0, k_theta=3.0)),
        ("LQR", UnicycleLQRController(v_nom=0.5)),
    ]

    results = []

    for name, controller in controllers:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("=" * 60)

        result = run_controller_test(model, controller, x0, target, T)
        results.append((name, result))

        # Performance metrics
        final_pos_error = np.linalg.norm(result.states[-1, :2] - target[:2])
        final_heading_error = abs(np.arctan2(
            np.sin(result.states[-1, 2] - target[2]),
            np.cos(result.states[-1, 2] - target[2])
        ))

        print(f"  Final position error: {final_pos_error:.4f}")
        print(f"  Final heading error: {np.degrees(final_heading_error):.2f} deg")
        print(f"  Final Lyapunov: {result.lyapunov_values[-1]:.6f}")

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print("=" * 60)

    # Trajectory comparison
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['blue', 'red', 'green', 'purple']

    for (name, result), color in zip(results, colors):
        ax.plot(result.states[:, 0], result.states[:, 1], '-',
                color=color, linewidth=1.5, label=name)

    ax.plot(x0[0], x0[1], 'ko', markersize=10, label='Start')
    ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Unicycle Trajectory Comparison')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, 'unicycle_trajectory_comparison')
    plt.close(fig)
    print("  Saved: unicycle_trajectory_comparison.png")

    # Lyapunov comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, result), color in zip(results, colors):
        V = result.lyapunov_values
        V_positive = np.maximum(V, 1e-10)
        ax.semilogy(result.time, V_positive, '-', color=color, linewidth=1.5, label=name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('V(x) [log scale]')
    ax.set_title('Unicycle Lyapunov Function Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, 'unicycle_lyapunov_comparison')
    plt.close(fig)
    print("  Saved: unicycle_lyapunov_comparison.png")

    # Error comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, result), color in zip(results, colors):
        errors = result.error_norms
        ax.semilogy(result.time, errors + 1e-10, '-', color=color, linewidth=1.5, label=name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error norm')
    ax.set_title('Unicycle Error Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, 'unicycle_error_comparison')
    plt.close(fig)
    print("  Saved: unicycle_error_comparison.png")

    # Heading evolution
    fig, ax = plt.subplots(figsize=(10, 4))
    for (name, result), color in zip(results, colors):
        ax.plot(result.time, np.degrees(result.states[:, 2]), '-',
                color=color, linewidth=1.5, label=name)

    ax.axhline(np.degrees(target[2]), color='r', linestyle='--', label='Target heading')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (degrees)')
    ax.set_title('Unicycle Heading Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, 'unicycle_heading')
    plt.close(fig)
    print("  Saved: unicycle_heading.png")

    print(f"\n{'='*60}")
    print("UNICYCLE SMOKE TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
