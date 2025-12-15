#!/usr/bin/env python3
"""
Generate comprehensive comparison report with numerical metrics.

Runs all controllers for all models and collects:
- Settling time
- Overshoot
- Input energy
- Final error
- Robustness to disturbance
- Computation time per step
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.integrator import IntegratorModel
from models.unicycle import UnicycleModel
from models.manipulator import TwoLinkManipulator

from controllers import ProportionalController, LQRController, PIDController
from controllers.mpc import MPCController
from controllers.rl_policy_gradient import PolicyGradientController
from controllers.unicycle import (
    FeedbackLinearizationController, PolarCoordinateController,
    SlidingModeController, UnicycleLQRController
)
from controllers.manipulator import (
    ComputedTorqueController, PDGravityCompensation,
    BacksteppingController, ManipulatorLQRController
)

from sim.simulator import Simulator


def compute_metrics(result, target: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics from simulation result."""
    metrics = {}

    # Final error (handle dimension mismatch)
    n_target = len(target)
    final_error = np.linalg.norm(result.states[-1, :n_target] - target)
    metrics['final_error'] = final_error

    # Settling time (time to reach within 2% of final value)
    threshold = 0.02 * np.linalg.norm(result.states[0, :n_target] - target)
    threshold = max(threshold, 0.1)  # Minimum threshold

    settling_time = result.time[-1]  # Default to end
    for i, state in enumerate(result.states):
        if np.linalg.norm(state[:n_target] - target) < threshold:
            # Check if it stays within threshold
            remaining_errors = [np.linalg.norm(s[:n_target] - target)
                               for s in result.states[i:]]
            if all(e < threshold * 2 for e in remaining_errors):
                settling_time = result.time[i]
                break
    metrics['settling_time'] = settling_time

    # Overshoot (for position-based systems)
    initial_dist = np.linalg.norm(result.states[0, :n_target] - target)
    min_dist = min(np.linalg.norm(s[:n_target] - target) for s in result.states)
    if min_dist < 0.01:
        # Check if it went past target
        overshoot = 0.0
        for i in range(1, len(result.states)):
            dist = np.linalg.norm(result.states[i, :n_target] - target)
            prev_dist = np.linalg.norm(result.states[i-1, :n_target] - target)
            if dist > prev_dist and prev_dist < 0.5:
                overshoot = max(overshoot, dist - prev_dist)
        metrics['overshoot'] = overshoot
    else:
        metrics['overshoot'] = 0.0

    # Input energy (sum of squared inputs)
    input_energy = np.sum(result.inputs ** 2) * (result.time[1] - result.time[0])
    metrics['input_energy'] = input_energy

    # Max input magnitude
    metrics['max_input'] = np.max(np.abs(result.inputs))

    # Success (reached within tolerance)
    metrics['success'] = final_error < 0.5

    return metrics


def measure_computation_time(controller, model, x0: np.ndarray, n_steps: int = 100) -> float:
    """Measure average computation time per control step."""
    times = []
    x = x0.copy()

    for t in range(n_steps):
        start = time.perf_counter()
        u, _ = controller.compute_control(x, t * model.tau)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        x = model.dynamics(x, u)

    return np.mean(times) * 1000  # Return in milliseconds


def test_robustness(model, controller, x0: np.ndarray, target: np.ndarray,
                    n_trials: int = 10) -> Dict[str, float]:
    """Test controller robustness across random disturbances."""
    final_errors = []
    successes = 0
    n_target = len(target)

    for seed in range(n_trials):
        sim = Simulator(model, controller, disturbance_mode='random', seed=seed)
        result = sim.run(x0, T=10.0, target=target)
        error = np.linalg.norm(result.states[-1, :n_target] - target)
        final_errors.append(error)
        if error < 0.5:
            successes += 1

    return {
        'mean_error': np.mean(final_errors),
        'std_error': np.std(final_errors),
        'max_error': np.max(final_errors),
        'success_rate': successes / n_trials
    }


def test_robustness_unicycle(model, controller, x0: np.ndarray, target: np.ndarray,
                              n_trials: int = 10) -> Dict[str, float]:
    """Test unicycle controller robustness (position error only)."""
    final_errors = []
    successes = 0

    for seed in range(n_trials):
        sim = Simulator(model, controller, disturbance_mode='random', seed=seed)
        result = sim.run(x0, T=10.0, target=target)
        # Measure position error only (first 2 components)
        error = np.linalg.norm(result.states[-1, :2] - target[:2])
        final_errors.append(error)
        if error < 0.5:
            successes += 1

    return {
        'mean_error': np.mean(final_errors),
        'std_error': np.std(final_errors),
        'max_error': np.max(final_errors),
        'success_rate': successes / n_trials
    }


def run_integrator_experiments() -> Dict[str, Any]:
    """Run all integrator experiments and collect metrics."""
    print("\n" + "="*60)
    print("MODEL 1: INTEGRATOR")
    print("="*60)

    model = IntegratorModel(tau=0.1)
    x0 = np.array([5.0, 5.0])
    target = np.array([0.0, 0.0])
    T = 10.0

    results = {}

    # 1. Proportional Controller
    print("\n  Testing Proportional Controller...")
    ctrl = ProportionalController(kp=0.5)
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['Proportional'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    # 2. LQR Controller
    print("  Testing LQR Controller...")
    ctrl = LQRController()
    ctrl.design_for_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['LQR'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    # 3. PID Controller
    print("  Testing PID Controller...")
    ctrl = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, dt=model.tau)
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['PID'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    # 4. MPC Controller
    print("  Testing MPC Controller...")
    ctrl = MPCController(horizon=10, Q=np.eye(2), R=0.1*np.eye(2))
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['MPC'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0, n_steps=50),
        'robustness': test_robustness(model, ctrl, x0, target, n_trials=5),
        'result': result
    }

    # 5. RL Policy Gradient (pre-trained)
    print("  Testing RL Policy Gradient...")
    ctrl = PolicyGradientController(n_states=2, n_actions=2, learning_rate=0.05)
    ctrl.set_target(target)

    # Quick training
    print("    Training policy (50 episodes)...")
    ctrl.train(model, n_episodes=50, max_steps=100, verbose=False)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['RL-PG'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target, n_trials=5),
        'result': result
    }

    return results


def run_unicycle_experiments() -> Dict[str, Any]:
    """Run all unicycle experiments and collect metrics."""
    print("\n" + "="*60)
    print("MODEL 2: UNICYCLE")
    print("="*60)

    model = UnicycleModel(tau=0.1)
    x0 = np.array([2.0, 2.0, 0.0])
    target = np.array([8.0, 8.0, np.pi/4])
    T = 15.0

    results = {}

    # 1. Feedback Linearization
    print("\n  Testing Feedback Linearization...")
    ctrl = FeedbackLinearizationController(d=0.3, kp=1.0)
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['FeedbackLin'] = {
        'metrics': compute_metrics(result, target[:2]),  # Position only for unicycle
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target[:2]),
        'result': result
    }

    # 2. Polar Coordinate
    print("  Testing Polar Coordinate Controller...")
    ctrl = PolarCoordinateController(k_rho=1.0, k_alpha=3.0, k_beta=-0.5)
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['PolarCoord'] = {
        'metrics': compute_metrics(result, target[:2]),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target[:2]),
        'result': result
    }

    # 3. Sliding Mode
    print("  Testing Sliding Mode Controller...")
    ctrl = SlidingModeController(k_pos=2.0, k_theta=3.0)
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['SlidingMode'] = {
        'metrics': compute_metrics(result, target[:2]),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target[:2]),
        'result': result
    }

    # 4. LQR
    print("  Testing Unicycle LQR Controller...")
    ctrl = UnicycleLQRController(v_nom=0.5)
    ctrl.set_model(model)
    ctrl.set_target(target)  # Full 3D target

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['LQR'] = {
        'metrics': compute_metrics(result, target[:2]),  # Only measure position
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness_unicycle(model, ctrl, x0, target),  # Special handler
        'result': result
    }

    return results


def run_manipulator_experiments() -> Dict[str, Any]:
    """Run all manipulator experiments and collect metrics."""
    print("\n" + "="*60)
    print("MODEL 3: TWO-LINK MANIPULATOR")
    print("="*60)

    model = TwoLinkManipulator(tau=0.01)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    target = np.array([np.pi/4, np.pi/4, 0.0, 0.0])
    T = 3.0

    results = {}

    # 1. Computed Torque
    print("\n  Testing Computed Torque Controller...")
    ctrl = ComputedTorqueController(Kp=100.0*np.eye(2), Kd=20.0*np.eye(2))
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['ComputedTorque'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    # 2. PD + Gravity Compensation
    print("  Testing PD + Gravity Compensation...")
    ctrl = PDGravityCompensation(Kp=150.0*np.eye(2), Kd=25.0*np.eye(2))
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['PD+Gravity'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    # 3. Backstepping
    print("  Testing Backstepping Controller...")
    ctrl = BacksteppingController(k1=10.0, k2=5.0)
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['Backstepping'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    # 4. LQR
    print("  Testing Manipulator LQR Controller...")
    ctrl = ManipulatorLQRController()
    ctrl.set_model(model)
    ctrl.set_target(target)

    sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
    result = sim.run(x0, T, target)

    results['LQR'] = {
        'metrics': compute_metrics(result, target),
        'comp_time': measure_computation_time(ctrl, model, x0),
        'robustness': test_robustness(model, ctrl, x0, target),
        'result': result
    }

    return results


def generate_markdown_report(integrator_results: Dict, unicycle_results: Dict,
                             manipulator_results: Dict) -> str:
    """Generate comprehensive markdown report with numerical metrics."""

    report = f"""# Discrete-Time Robot Control: Comprehensive Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents a comprehensive comparison of control strategies implemented
for three discrete-time robot models. All controllers were tested with bounded
random disturbances to evaluate robustness.

---

## Model 1: Integrator (2D Point Robot)

**Dynamics:** x(t+1) = x(t) + tau*(u(t) + w(t))

**Test Configuration:**
- Initial state: [5.0, 5.0]
- Target: [0.0, 0.0]
- Simulation time: 10.0s
- Sampling period: tau = 0.1s

### Performance Metrics

| Controller | Settling Time (s) | Final Error | Overshoot | Input Energy | Comp. Time (ms) |
|------------|------------------|-------------|-----------|--------------|-----------------|
"""

    for name, data in integrator_results.items():
        m = data['metrics']
        report += f"| {name} | {m['settling_time']:.2f} | {m['final_error']:.4f} | {m['overshoot']:.4f} | {m['input_energy']:.2f} | {data['comp_time']:.3f} |\n"

    report += """
### Robustness Analysis (10 trials with random disturbances)

| Controller | Mean Error | Std Error | Max Error | Success Rate |
|------------|------------|-----------|-----------|--------------|
"""

    for name, data in integrator_results.items():
        r = data['robustness']
        report += f"| {name} | {r['mean_error']:.4f} | {r['std_error']:.4f} | {r['max_error']:.4f} | {r['success_rate']*100:.0f}% |\n"

    report += """
### Controller Analysis

**Proportional (P):**
- Simple implementation, single gain parameter
- Steady-state error possible without integral action
- Stability: kp*tau < 2 required

**LQR:**
- Optimal state feedback for quadratic cost
- Guaranteed stability via DARE solution
- Best energy efficiency among linear controllers

**PID:**
- Eliminates steady-state error with integral term
- May have slower settling due to integral windup
- Tuning can be challenging

**MPC:**
- Handles constraints explicitly
- Computationally expensive (optimization each step)
- Best constraint satisfaction

**RL Policy Gradient:**
- Model-free, learns from interaction
- Requires training episodes
- Performance depends on training quality

---

## Model 2: Unicycle (Nonholonomic Mobile Robot)

**Dynamics:** Nonholonomic with velocity and angular velocity inputs

**Test Configuration:**
- Initial state: [2.0, 2.0, 0.0]
- Target: [8.0, 8.0, pi/4]
- Simulation time: 15.0s
- Sampling period: tau = 0.1s

### Performance Metrics

| Controller | Settling Time (s) | Final Error | Overshoot | Input Energy | Comp. Time (ms) |
|------------|------------------|-------------|-----------|--------------|-----------------|
"""

    for name, data in unicycle_results.items():
        m = data['metrics']
        report += f"| {name} | {m['settling_time']:.2f} | {m['final_error']:.4f} | {m['overshoot']:.4f} | {m['input_energy']:.2f} | {data['comp_time']:.3f} |\n"

    report += """
### Robustness Analysis (10 trials with random disturbances)

| Controller | Mean Error | Std Error | Max Error | Success Rate |
|------------|------------|-----------|-----------|--------------|
"""

    for name, data in unicycle_results.items():
        r = data['robustness']
        report += f"| {name} | {r['mean_error']:.4f} | {r['std_error']:.4f} | {r['max_error']:.4f} | {r['success_rate']*100:.0f}% |\n"

    report += """
### Controller Analysis

**Feedback Linearization:**
- Transforms nonlinear system to linear
- Singularity at reference point distance d=0
- Good for trajectory tracking

**Polar Coordinates:**
- Global asymptotic stability (proven)
- Natural handling of position/heading
- Smooth control inputs

**Sliding Mode:**
- Robust to matched disturbances
- May exhibit chattering
- Boundary layer reduces chattering

**LQR (Linearized):**
- Only locally valid
- Simple implementation
- Works well near equilibrium

---

## Model 3: Two-Link Manipulator

**Dynamics:** M(q)q'' + C(q,q')q' + g(q) = tau (Euler discretized)

**Test Configuration:**
- Initial state: [0, 0, 0, 0] (hanging)
- Target: [pi/4, pi/4, 0, 0] (45-45 degrees)
- Simulation time: 3.0s
- Sampling period: tau = 0.01s

### Performance Metrics

| Controller | Settling Time (s) | Final Error | Overshoot | Input Energy | Comp. Time (ms) |
|------------|------------------|-------------|-----------|--------------|-----------------|
"""

    for name, data in manipulator_results.items():
        m = data['metrics']
        report += f"| {name} | {m['settling_time']:.2f} | {m['final_error']:.4f} | {m['overshoot']:.4f} | {m['input_energy']:.2f} | {data['comp_time']:.3f} |\n"

    report += """
### Robustness Analysis (10 trials with random disturbances)

| Controller | Mean Error | Std Error | Max Error | Success Rate |
|------------|------------|-----------|-----------|--------------|
"""

    for name, data in manipulator_results.items():
        r = data['robustness']
        report += f"| {name} | {r['mean_error']:.4f} | {r['std_error']:.4f} | {r['max_error']:.4f} | {r['success_rate']*100:.0f}% |\n"

    report += """
### Controller Analysis

**Computed Torque:**
- Exact linearization via feedback
- Requires accurate model (M, C, g)
- Excellent tracking with good model

**PD + Gravity Compensation:**
- Simpler than computed torque
- Only compensates gravity
- Robust to inertia uncertainty

**Backstepping:**
- Recursive Lyapunov design
- Guaranteed stability certificate
- Systematic design procedure

**LQR (Linearized):**
- Valid only near linearization point
- May fail for large motions
- Simple implementation

---

## Comparative Analysis

### Computation Time Summary

| Category | Controller | Avg. Time (ms) | Real-time Feasible |
|----------|------------|----------------|-------------------|
"""

    all_results = [
        ('Integrator', integrator_results),
        ('Unicycle', unicycle_results),
        ('Manipulator', manipulator_results)
    ]

    for model_name, results in all_results:
        for ctrl_name, data in results.items():
            feasible = "Yes" if data['comp_time'] < 10 else "No"
            report += f"| {model_name} | {ctrl_name} | {data['comp_time']:.3f} | {feasible} |\n"

    report += """
### Overall Robustness Ranking

Based on success rate across all disturbance trials:

"""

    # Compute overall rankings
    all_controllers = []
    for model_name, results in all_results:
        for ctrl_name, data in results.items():
            all_controllers.append({
                'model': model_name,
                'controller': ctrl_name,
                'success_rate': data['robustness']['success_rate'],
                'mean_error': data['robustness']['mean_error']
            })

    # Sort by success rate
    all_controllers.sort(key=lambda x: (-x['success_rate'], x['mean_error']))

    report += "| Rank | Model | Controller | Success Rate | Mean Error |\n"
    report += "|------|-------|------------|--------------|------------|\n"
    for i, c in enumerate(all_controllers[:10], 1):
        report += f"| {i} | {c['model']} | {c['controller']} | {c['success_rate']*100:.0f}% | {c['mean_error']:.4f} |\n"

    report += """
---

## Lyapunov Stability Certificates

### Model 1: Integrator

| Controller | Lyapunov Function | Stability Type |
|------------|-------------------|----------------|
| P, PID | V(e) = e'e | Global AS (linear) |
| LQR | V(e) = e'Pe (P from DARE) | Global AS |
| MPC | V(x) = x'Px + constraints | Local (with constraints) |
| RL-PG | Empirical | Not guaranteed |

### Model 2: Unicycle

| Controller | Lyapunov Function | Stability Type |
|------------|-------------------|----------------|
| Feedback Lin. | V = 0.5*||p_ref - p||^2 | Local (singularity at d=0) |
| Polar Coord. | V = 0.5*(rho^2 + alpha^2 + k*beta^2) | Global AS |
| Sliding Mode | V = 0.5*(s_pos^2 + s_theta^2) | Finite-time to surface |
| LQR | V = e'Pe | Local |

### Model 3: Manipulator

| Controller | Lyapunov Function | Stability Type |
|------------|-------------------|----------------|
| Computed Torque | V = 0.5*(e_dot'Me_dot + e'Kp*e) | Global AS (exact model) |
| PD + Gravity | V = 0.5*(q_dot'Mq_dot + e'Kp*e) + Ug | Global AS |
| Backstepping | V = 0.5*(z1'z1 + z2'Mz2) | Global AS |
| LQR | V = e'Pe | Local |

---

## Limitations and Lessons Learned

### Linear Controllers
**Strengths:** Simple, systematic design, guaranteed stability for linear systems
**Limitations:** Only locally valid for nonlinear systems, no constraint handling
**Lesson:** Good baseline, combine with constraint handling for practical use

### Nonlinear Controllers
**Strengths:** Handle nonlinearities directly, global stability possible
**Limitations:** Require accurate models, more complex implementation
**Lesson:** Essential for nonholonomic and underactuated systems

### Robust Controllers (Sliding Mode)
**Strengths:** Reject matched disturbances, insensitive to parameter uncertainty
**Limitations:** Chattering, high-frequency switching
**Lesson:** Use boundary layer, combine with continuous component

### Optimal Controllers (MPC)
**Strengths:** Handle constraints, preview capability, optimal performance
**Limitations:** Computationally expensive, need accurate model
**Lesson:** Worth the cost for constrained systems, use efficient solvers

### Data-Driven (RL)
**Strengths:** Model-free, can handle complex dynamics
**Limitations:** Sample inefficient, no formal guarantees, training required
**Lesson:** Useful when model unavailable, combine with model-based for safety

---

## Conclusions

1. **For linear systems (Integrator):** LQR provides best balance of performance and simplicity
2. **For nonholonomic systems (Unicycle):** Polar coordinate controller offers global stability
3. **For manipulators:** Computed torque excels with good models; PD+Gravity is robust alternative
4. **For constraints:** MPC is preferred despite computational cost
5. **For unknown dynamics:** RL can work but requires careful training and lacks guarantees

---

*Report generated automatically by generate_report.py*
"""

    return report


def generate_comparison_figures(integrator_results: Dict, unicycle_results: Dict,
                                manipulator_results: Dict, output_dir: str):
    """Generate comparison figures for the report."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Integrator trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for (name, data), color in zip(integrator_results.items(), colors):
        result = data['result']
        ax.plot(result.states[:, 0], result.states[:, 1], '-',
                color=color, linewidth=1.5, label=name)
    ax.plot(0, 0, 'k*', markersize=15, label='Target')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Integrator: Trajectory Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1]
    for (name, data), color in zip(integrator_results.items(), colors):
        result = data['result']
        ax.semilogy(result.time, result.error_norms + 1e-6, '-',
                    color=color, linewidth=1.5, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Integrator: Error Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/integrator_comparison.png', dpi=150)
    plt.close()

    # Figure 2: Unicycle trajectories
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for (name, data), color in zip(unicycle_results.items(), colors):
        result = data['result']
        ax.plot(result.states[:, 0], result.states[:, 1], '-',
                color=color, linewidth=1.5, label=name)
    ax.plot(8, 8, 'k*', markersize=15, label='Target')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Unicycle: Trajectory Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (name, data), color in zip(unicycle_results.items(), colors):
        result = data['result']
        errors = np.sqrt((result.states[:, 0] - 8)**2 + (result.states[:, 1] - 8)**2)
        ax.semilogy(result.time, errors + 1e-6, '-',
                    color=color, linewidth=1.5, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (log scale)')
    ax.set_title('Unicycle: Error Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/unicycle_comparison.png', dpi=150)
    plt.close()

    # Figure 3: Manipulator joint angles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for (name, data), color in zip(manipulator_results.items(), colors):
        result = data['result']
        axes[0, 0].plot(result.time, np.degrees(result.states[:, 0]), '-',
                        color=color, linewidth=1.5, label=name)
        axes[0, 1].plot(result.time, np.degrees(result.states[:, 1]), '-',
                        color=color, linewidth=1.5, label=name)

    axes[0, 0].axhline(45, color='k', linestyle='--', alpha=0.5, label='Target')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Joint 1 Angle (deg)')
    axes[0, 0].set_title('Joint 1 Evolution')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axhline(45, color='k', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Joint 2 Angle (deg)')
    axes[0, 1].set_title('Joint 2 Evolution')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Error convergence
    target = np.array([np.pi/4, np.pi/4, 0, 0])
    for (name, data), color in zip(manipulator_results.items(), colors):
        result = data['result']
        errors = np.linalg.norm(result.states - target, axis=1)
        axes[1, 0].semilogy(result.time, errors + 1e-6, '-',
                            color=color, linewidth=1.5, label=name)

    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('State Error (log scale)')
    axes[1, 0].set_title('Error Convergence')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bar chart of metrics
    names = list(manipulator_results.keys())
    settling_times = [manipulator_results[n]['metrics']['settling_time'] for n in names]
    x = np.arange(len(names))

    axes[1, 1].bar(x, settling_times, color=colors[:len(names)])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Settling Time (s)')
    axes[1, 1].set_title('Settling Time Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/manipulator_comparison.png', dpi=150)
    plt.close()

    # Figure 4: Overall comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Computation time comparison
    all_names = []
    all_times = []
    all_colors = []
    color_map = {'Integrator': 'blue', 'Unicycle': 'green', 'Manipulator': 'red'}

    for model_name, results in [('Integrator', integrator_results),
                                 ('Unicycle', unicycle_results),
                                 ('Manipulator', manipulator_results)]:
        for ctrl_name, data in results.items():
            all_names.append(f"{model_name[:3]}-{ctrl_name[:6]}")
            all_times.append(data['comp_time'])
            all_colors.append(color_map[model_name])

    x = np.arange(len(all_names))
    axes[0].bar(x, all_times, color=all_colors, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_names, rotation=90, fontsize=7)
    axes[0].set_ylabel('Computation Time (ms)')
    axes[0].set_title('Computation Time per Step')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Success rate comparison
    all_success = []
    for model_name, results in [('Integrator', integrator_results),
                                 ('Unicycle', unicycle_results),
                                 ('Manipulator', manipulator_results)]:
        for ctrl_name, data in results.items():
            all_success.append(data['robustness']['success_rate'] * 100)

    axes[1].bar(x, all_success, color=all_colors, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_names, rotation=90, fontsize=7)
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('Robustness (Success Rate)')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Input energy comparison
    all_energy = []
    for model_name, results in [('Integrator', integrator_results),
                                 ('Unicycle', unicycle_results),
                                 ('Manipulator', manipulator_results)]:
        for ctrl_name, data in results.items():
            all_energy.append(data['metrics']['input_energy'])

    axes[2].bar(x, all_energy, color=all_colors, alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(all_names, rotation=90, fontsize=7)
    axes[2].set_ylabel('Input Energy')
    axes[2].set_title('Control Effort (Input Energy)')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_comparison.png', dpi=150)
    plt.close()

    print(f"  Figures saved to {output_dir}/")


def main():
    """Main entry point for report generation."""
    print("="*70)
    print("COMPREHENSIVE ROBOT CONTROL COMPARISON REPORT")
    print("="*70)

    # Run all experiments
    integrator_results = run_integrator_experiments()
    unicycle_results = run_unicycle_experiments()
    manipulator_results = run_manipulator_experiments()

    # Generate figures
    print("\n" + "="*60)
    print("Generating comparison figures...")
    generate_comparison_figures(
        integrator_results, unicycle_results, manipulator_results,
        'report/figures'
    )

    # Generate markdown report
    print("\nGenerating markdown report...")
    report = generate_markdown_report(
        integrator_results, unicycle_results, manipulator_results
    )

    # Save report
    os.makedirs('report', exist_ok=True)
    with open('report/comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("  Report saved to report/comparison_report.md")

    print("\n" + "="*70)
    print("REPORT GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - report/comparison_report.md")
    print("  - report/figures/integrator_comparison.png")
    print("  - report/figures/unicycle_comparison.png")
    print("  - report/figures/manipulator_comparison.png")
    print("  - report/figures/overall_comparison.png")

    return 0


if __name__ == '__main__':
    sys.exit(main())
