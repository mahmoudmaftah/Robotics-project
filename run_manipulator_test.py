#!/usr/bin/env python3
"""
Smoke test for Model 3 (Two-Link Manipulator).

Tests all manipulator controllers:
(a) Computed Torque (Inverse Dynamics)
(b) PD + Gravity Compensation
(c) Backstepping
(d) LQR (linearized)

Based on discrete-time manipulator from Symbolic_control_lecture-7.pdf:
    x(t+1) = x(t) + tau * f(x(t), u(t))
    where x = [theta, theta_dot], f includes M^{-1}(tau - c - g)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.manipulator import TwoLinkManipulator
from controllers.manipulator import (
    ComputedTorqueController,
    PDGravityCompensation,
    BacksteppingController,
    ManipulatorLQRController
)
from sim.simulator import Simulator
from sim.plotting import save_figure


def ensure_dirs():
    """Create output directories."""
    os.makedirs('report/figures', exist_ok=True)


def run_controller_test(model, controller, x0, target, T=5.0):
    """Run a single controller test."""
    controller.set_model(model)
    controller.set_target(target)
    controller.reset()

    sim = Simulator(model, controller, disturbance_mode='random', seed=42)
    result = sim.run(x0, T)

    return result


def visualize_manipulator(ax, model, theta, color='blue', alpha=1.0):
    """Draw manipulator configuration."""
    p1, p2 = model.forward_kinematics(theta)

    # Base
    ax.plot(0, 0, 'ko', markersize=10)

    # Links
    ax.plot([0, p1[0]], [0, p1[1]], '-', color=color, linewidth=3, alpha=alpha)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, linewidth=3, alpha=alpha)

    # Joints
    ax.plot(p1[0], p1[1], 'o', color=color, markersize=6, alpha=alpha)
    ax.plot(p2[0], p2[1], 's', color=color, markersize=8, alpha=alpha)


def main():
    """Run manipulator smoke test."""
    print("=" * 60)
    print("SMOKE TEST: Model 3 (Two-Link Manipulator)")
    print("=" * 60)

    ensure_dirs()

    # Model setup (smaller time step for stability)
    model = TwoLinkManipulator(tau=0.01)
    print(f"\nModel: {model}")
    print(f"  Sampling period: tau = {model.tau}")
    print(f"  Link masses: m1={model.m1}, m2={model.m2}")
    print(f"  Link lengths: l1={model.l1}, l2={model.l2}")

    # Test configuration
    # Start: arm hanging down (theta1=theta2=0 means pointing right)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # [theta1, theta2, dtheta1, dtheta2]
    # Target: arm reaching up
    target = np.array([np.pi/3, np.pi/4, 0.0, 0.0])  # 60 deg, 45 deg
    T = 3.0

    print(f"\nTest configuration:")
    print(f"  Initial state: q0 = [{np.degrees(x0[0]):.1f}, {np.degrees(x0[1]):.1f}] deg")
    print(f"  Target state: q* = [{np.degrees(target[0]):.1f}, {np.degrees(target[1]):.1f}] deg")
    print(f"  Simulation time: T = {T} s")

    # Controllers to test
    controllers = [
        ("Computed Torque", ComputedTorqueController(
            Kp=100*np.eye(2), Kd=20*np.eye(2))),
        ("PD+Gravity", PDGravityCompensation(
            Kp=80*np.eye(2), Kd=15*np.eye(2))),
        ("Backstepping", BacksteppingController(k1=8.0, k2=15.0)),
        ("LQR", ManipulatorLQRController()),
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
        final_vel = np.linalg.norm(result.states[-1, 2:])

        print(f"  Final position error: {np.degrees(final_pos_error):.4f} deg")
        print(f"  Final velocity norm: {final_vel:.4f} rad/s")
        print(f"  Final Lyapunov: {result.lyapunov_values[-1]:.6f}")

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print("=" * 60)

    colors = ['blue', 'red', 'green', 'purple']

    # Joint angle evolution
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for (name, result), color in zip(results, colors):
        axes[0].plot(result.time, np.degrees(result.states[:, 0]), '-',
                     color=color, linewidth=1.5, label=name)
        axes[1].plot(result.time, np.degrees(result.states[:, 1]), '-',
                     color=color, linewidth=1.5, label=name)

    axes[0].axhline(np.degrees(target[0]), color='k', linestyle='--', label='Target')
    axes[1].axhline(np.degrees(target[1]), color='k', linestyle='--', label='Target')

    axes[0].set_ylabel('$\\theta_1$ (deg)')
    axes[1].set_ylabel('$\\theta_2$ (deg)')
    axes[1].set_xlabel('Time (s)')
    axes[0].set_title('Manipulator Joint Angles')
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, 'manipulator_joint_angles')
    plt.close(fig)
    print("  Saved: manipulator_joint_angles.png")

    # Lyapunov comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, result), color in zip(results, colors):
        V = result.lyapunov_values
        V_positive = np.maximum(V, 1e-10)
        ax.semilogy(result.time, V_positive, '-', color=color, linewidth=1.5, label=name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('V(x) [log scale]')
    ax.set_title('Manipulator Lyapunov Function Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, 'manipulator_lyapunov_comparison')
    plt.close(fig)
    print("  Saved: manipulator_lyapunov_comparison.png")

    # Error comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, result), color in zip(results, colors):
        errors = result.error_norms
        ax.semilogy(result.time, errors + 1e-10, '-', color=color, linewidth=1.5, label=name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error norm (rad)')
    ax.set_title('Manipulator Position Error Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, 'manipulator_error_comparison')
    plt.close(fig)
    print("  Saved: manipulator_error_comparison.png")

    # Manipulator configurations visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for idx, ((name, result), ax) in enumerate(zip(results, axes.flatten())):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(name)

        # Draw trajectory of end-effector
        ee_traj = []
        for state in result.states[::10]:  # Subsample
            _, p2 = model.forward_kinematics(state[:2])
            ee_traj.append(p2)
        ee_traj = np.array(ee_traj)
        ax.plot(ee_traj[:, 0], ee_traj[:, 1], 'b-', alpha=0.3, linewidth=1)

        # Draw initial, middle, and final configurations
        n = len(result.states)
        for i, alpha_val in [(0, 0.3), (n//2, 0.6), (-1, 1.0)]:
            visualize_manipulator(ax, model, result.states[i, :2],
                                  color='blue', alpha=alpha_val)

        # Target configuration
        visualize_manipulator(ax, model, target[:2], color='red', alpha=0.5)

    fig.suptitle('Manipulator Motion Sequences', fontsize=14)
    fig.tight_layout()
    save_figure(fig, 'manipulator_configurations')
    plt.close(fig)
    print("  Saved: manipulator_configurations.png")

    # Control input comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for (name, result), color in zip(results, colors):
        axes[0].plot(result.time, result.inputs[:, 0], '-',
                     color=color, linewidth=1, label=name, alpha=0.7)
        axes[1].plot(result.time, result.inputs[:, 1], '-',
                     color=color, linewidth=1, label=name, alpha=0.7)

    axes[0].set_ylabel('$\\tau_1$ (Nm)')
    axes[1].set_ylabel('$\\tau_2$ (Nm)')
    axes[1].set_xlabel('Time (s)')
    axes[0].set_title('Manipulator Control Torques')
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, 'manipulator_torques')
    plt.close(fig)
    print("  Saved: manipulator_torques.png")

    print(f"\n{'='*60}")
    print("MANIPULATOR SMOKE TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
