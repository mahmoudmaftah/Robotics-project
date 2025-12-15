# Robotics Control Framework - Comprehensive Usage Guide

A Python framework for discrete-time robot control with multiple controller families, Lyapunov stability analysis, and symbolic control capabilities.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Robot Models](#robot-models)
4. [Controllers](#controllers)
5. [Simulation Engine](#simulation-engine)
6. [Symbolic Control](#symbolic-control)
7. [Interactive Demo](#interactive-demo)
8. [Jupyter Notebooks](#jupyter-notebooks)
9. [Animation Export](#animation-export)
10. [Testing](#testing)
11. [Examples](#examples)

---

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

Optional dependencies:
```bash
pip install pillow    # For GIF export
pip install ffmpeg-python  # For MP4 export (also need ffmpeg binary)
pip install jupyter   # For notebook demos
```

### Quick Start

```python
import numpy as np
from models.integrator import IntegratorModel
from controllers import ProportionalController
from sim.simulator import Simulator

# Create model and controller
model = IntegratorModel(tau=0.1)
controller = ProportionalController(kp=0.5)
controller.set_model(model)
controller.set_target(np.array([0.0, 0.0]))

# Run simulation
sim = Simulator(model, controller)
result = sim.run(x0=np.array([5.0, 5.0]), T=10.0)

# Plot results
result.plot()
```

---

## Project Structure

```
Robotics/
├── models/                    # Robot dynamic models
│   ├── base.py               # Base model class
│   ├── integrator.py         # 2D integrator
│   ├── unicycle.py           # Nonholonomic mobile robot
│   └── manipulator.py        # Two-link planar arm
│
├── controllers/               # Control algorithms
│   ├── base.py               # Base controller class
│   ├── proportional.py       # P controller
│   ├── pid.py                # PID controller
│   ├── lqr.py                # Linear Quadratic Regulator
│   ├── mpc.py                # Model Predictive Control
│   ├── rl_policy_gradient.py # REINFORCE algorithm
│   ├── unicycle/             # Unicycle-specific controllers
│   │   ├── feedback_linearization.py
│   │   ├── polar_controller.py
│   │   ├── sliding_mode.py
│   │   └── unicycle_lqr.py
│   └── manipulator/          # Manipulator-specific controllers
│       ├── computed_torque.py
│       ├── pd_gravity.py
│       ├── backstepping.py
│       └── manipulator_lqr.py
│
├── sim/                       # Simulation engine
│   ├── simulator.py          # Core simulator
│   ├── plotting.py           # Visualization utilities
│   └── animation.py          # GIF/MP4 export
│
├── symbolic/                  # Symbolic control
│   ├── grid_abstraction.py   # Space discretization
│   ├── reach_avoid.py        # Reach-avoid planning
│   └── ltl_automata.py       # LTL/Buchi automata for patrolling
│
├── analysis/                  # Stability analysis
│   └── lyapunov.py           # Lyapunov function computation
│
├── notebooks/                 # Jupyter demos
│   ├── integrator_demo.ipynb
│   ├── unicycle_demo.ipynb
│   └── manipulator_demo.ipynb
│
├── tests/                     # Unit tests
│   ├── test_models.py
│   ├── test_controllers.py
│   └── test_mpc_rl.py
│
├── interactive_demo.py        # Interactive CLI demo
├── run_smoke_test.py          # Quick verification script
└── USAGE_GUIDE.md            # This file
```

---

## Robot Models

All models inherit from `RobotModel` and implement discrete-time dynamics:
```
x(t+1) = f(x(t), u(t), w(t))
```

### 1. Integrator Model

Simple 2D point mass with direct velocity control.

```python
from models.integrator import IntegratorModel

model = IntegratorModel(tau=0.1)  # tau = sampling period

# Dynamics: x(t+1) = x(t) + tau * (u(t) + w(t))
# State: [x, y] in [-10, 10]^2
# Input: [vx, vy] in [-1, 1]^2
# Disturbance: w in [-0.05, 0.05]^2
```

### 2. Unicycle Model

Nonholonomic mobile robot with velocity and angular velocity inputs.

```python
from models.unicycle import UnicycleModel

model = UnicycleModel(tau=0.1)

# Dynamics:
#   x(t+1) = x(t) + tau * v(t) * cos(theta(t))
#   y(t+1) = y(t) + tau * v(t) * sin(theta(t))
#   theta(t+1) = theta(t) + tau * omega(t)
#
# State: [x, y, theta]
# Input: [v, omega] with v in [0.25, 1], omega in [-1, 1]
```

### 3. Two-Link Manipulator

Planar robot arm with Lagrangian dynamics.

```python
from models.manipulator import TwoLinkManipulator

model = TwoLinkManipulator(tau=0.01, m1=1.0, m2=1.0, l1=0.5, l2=0.5)

# Dynamics: M(theta) * theta_ddot + C(theta, theta_dot) + g(theta) = tau
#
# State: [theta1, theta2, theta1_dot, theta2_dot]
# Input: [tau1, tau2] (joint torques) in [-10, 10]^2
```

**Useful methods:**
```python
# Get end-effector position
p1, p2 = model.forward_kinematics(theta)

# Get dynamics components
M = model.mass_matrix(theta)
c = model.coriolis_vector(theta, theta_dot)
g = model.gravity_vector(theta)
```

---

## Controllers

All controllers inherit from `Controller` and implement:
```python
def compute_control(self, x, t) -> Tuple[np.ndarray, Dict]:
    """Returns (control_input, diagnostic_info)"""
```

### Generic Controllers

#### Proportional Controller
```python
from controllers import ProportionalController

ctrl = ProportionalController(kp=0.5)
ctrl.set_model(model)
ctrl.set_target(target)

# Stability analysis
stability = ctrl.analyze_stability(tau=model.tau)
print(f"Stable: {stability['is_stable']}")
print(f"Eigenvalues: {stability['eigenvalues']}")
```

#### LQR Controller
```python
from controllers import LQRController

ctrl = LQRController(Q=np.eye(2), R=0.1*np.eye(2))
ctrl.design_for_model(model)
ctrl.set_target(target)

# Get Lyapunov matrix P (from DARE solution)
print(f"Lyapunov P:\n{ctrl.P}")
```

#### PID Controller
```python
from controllers import PIDController

ctrl = PIDController(kp=0.5, ki=0.1, kd=0.05)
ctrl.set_model(model)
ctrl.set_target(target)
```

#### MPC Controller
```python
from controllers import MPCController

ctrl = MPCController(
    horizon=10,
    Q=np.eye(2),          # State cost
    R=0.1 * np.eye(2),    # Input cost
    Q_terminal=None       # Optional terminal cost
)
ctrl.set_model(model)
ctrl.set_target(target)

# Analyze performance
perf = ctrl.analyze_performance()
print(f"Mean solve time: {perf['mean_solve_time']*1000:.2f} ms")
```

#### RL Policy Gradient
```python
from controllers import PolicyGradientController

ctrl = PolicyGradientController(
    n_states=2,
    n_actions=2,
    learning_rate=0.05,
    sigma=0.3
)
ctrl.set_target(target)

# Train the policy
returns = ctrl.train(model, n_episodes=100, max_steps=100, verbose=True)

# Use trained policy
u, info = ctrl.compute_control(state, t)
```

### Unicycle Controllers

```python
from controllers.unicycle import (
    FeedbackLinearizationController,
    PolarCoordinateController,
    SlidingModeController,
    UnicycleLQRController
)

# Feedback Linearization (reference point ahead)
ctrl = FeedbackLinearizationController(d=0.3, kp=1.0)

# Polar Coordinates (globally stable)
ctrl = PolarCoordinateController(k_rho=1.0, k_alpha=3.0, k_beta=-0.5)

# Sliding Mode (robust to disturbances)
ctrl = SlidingModeController(k_pos=2.0, k_theta=3.0)

# LQR (linearized around heading)
ctrl = UnicycleLQRController(v_nom=0.5)
```

### Manipulator Controllers

```python
from controllers.manipulator import (
    ComputedTorqueController,
    PDGravityCompensation,
    BacksteppingController,
    ManipulatorLQRController
)

# Computed Torque (exact linearization)
ctrl = ComputedTorqueController(kp=100.0, kd=20.0)

# PD + Gravity Compensation
ctrl = PDGravityCompensation(kp=150.0, kd=25.0)

# Backstepping (Lyapunov-based)
ctrl = BacksteppingController(k1=10.0, k2=5.0)

# LQR (linearized)
ctrl = ManipulatorLQRController()
```

---

## Simulation Engine

### Basic Simulation

```python
from sim.simulator import Simulator

sim = Simulator(
    model=model,
    controller=controller,
    disturbance_mode='random',  # 'none', 'constant', 'random'
    seed=42
)

result = sim.run(
    x0=initial_state,
    T=simulation_time,
    target=target_state  # Optional
)
```

### SimulationResult

```python
# Access simulation data
result.time           # Time array
result.states         # State trajectory (N x n_states)
result.inputs         # Control inputs (N x n_inputs)
result.error_norms    # Error norm at each step
result.lyapunov_values # Lyapunov function values

# Compute metrics
metrics = result.compute_metrics()
print(f"Settling time: {metrics['settling_time']}")
print(f"Input energy: {metrics['input_energy']}")

# Quick plot
result.plot()
```

### Plotting Utilities

```python
from sim.plotting import plot_trajectory, plot_phase_portrait, plot_lyapunov

# 2D trajectory plot
fig, ax = plot_trajectory(result.states, result.time)

# Phase portrait (for 2D systems)
fig, ax = plot_phase_portrait(result.states)

# Lyapunov function evolution
fig, ax = plot_lyapunov(result.time, result.lyapunov_values)
```

---

## Symbolic Control

### Grid Abstraction

```python
from symbolic.grid_abstraction import GridAbstraction

# Create grid discretization
abstraction = GridAbstraction(
    bounds=model.x_bounds,
    resolution=(20, 20),
    model=model
)

# Define goal region (polygon vertices)
goal = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
abstraction.set_goal_region(goal)

# Define obstacles
obstacles = [
    np.array([[-3, -3], [-1, -3], [-1, 1], [-3, 1]]),
    np.array([[2, 2], [4, 2], [4, 5], [2, 5]])
]
abstraction.set_obstacles(obstacles)

# Visualize
fig, ax = plt.subplots()
abstraction.visualize(ax)
```

### Reach-Avoid Planning

```python
from symbolic.reach_avoid import ReachAvoidPlanner, ReachAvoidController

# Create planner
planner = ReachAvoidPlanner(abstraction)
planner.compute_value_function()
planner.compute_policy()

# Create controller
ra_ctrl = ReachAvoidController(
    planner,
    waypoint_tolerance=0.5,
    kp=0.8
)
ra_ctrl.set_model(model)

# Simulate
sim = Simulator(model, ra_ctrl)
result = sim.run(x0, T=15.0)
```

### Recurrence/Patrolling (LTL)

```python
from symbolic.ltl_automata import RecurrenceController, create_patrol_regions

# Create patrol regions
regions = create_patrol_regions(model.x_bounds, n_regions=4)

# Or define custom regions
regions = {
    'A': np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
    'B': np.array([[5, 5], [7, 5], [7, 7], [5, 7]])
}

# Create recurrence controller
patrol_ctrl = RecurrenceController(
    abstraction,
    regions,
    spec_type='patrol',  # 'patrol' or 'sequence'
    kp=0.8
)
patrol_ctrl.set_model(model)

# Compute patrol cycle
success = patrol_ctrl.compute_patrol_cycle(x0)
if success:
    print(f"Patrol cycle length: {len(patrol_ctrl.cycle)}")
```

**Specification Types:**
- `'patrol'`: Visit all regions infinitely often (GF r1 & GF r2 & ...)
- `'sequence'`: Visit regions in order, repeat (F(r1 & F(r2 & F(r3 ...))))

---

## Interactive Demo

Run the interactive demonstration with user-configurable parameters:

```bash
python interactive_demo.py
```

**Features:**
- User input for number of obstacles, grid resolution, simulation parameters
- Random obstacle/goal/start configuration
- Comparison of multiple controllers
- Success rate statistics

**Sample session:**
```
=== Robotics Control - Interactive Demo ===

Enter number of obstacles (1-5) [default=2]: 3
Enter grid resolution (10-50) [default=20]: 25
Enter number of simulation runs (1-10) [default=3]: 5
Enter simulation time in seconds (5-30) [default=15]: 20
Enter random seed [default=random]: 42

Run 1/5:
  Generated 3 obstacles
  Start: [7.2, 1.5], Goal region at [3.1, 6.8]
  ...
```

---

## Jupyter Notebooks

Three interactive notebooks are provided in the `notebooks/` directory:

### 1. integrator_demo.ipynb

Demonstrates all controllers for the 2D integrator:
- Proportional, LQR, MPC, RL Policy Gradient
- Reach-Avoid with obstacles
- Stability analysis comparison

### 2. unicycle_demo.ipynb

Demonstrates unicycle controllers:
- Feedback Linearization
- Polar Coordinates
- Sliding Mode
- LQR

### 3. manipulator_demo.ipynb

Demonstrates two-link manipulator controllers:
- Computed Torque
- PD with Gravity Compensation
- Backstepping
- LQR
- Arm animation

**Running notebooks:**
```bash
cd notebooks
jupyter notebook
```

---

## Animation Export

Export simulation results as animations:

### Quick Export

```python
from sim.animation import export_simulation

# Export as GIF
export_simulation(result, model, 'output/trajectory', format='gif')

# Export as MP4 (requires ffmpeg)
export_simulation(result, model, 'output/trajectory', format='mp4')

# Export as HTML
export_simulation(result, model, 'output/trajectory', format='html')
```

### Custom Animation

```python
from sim.animation import AnimationExporter

exporter = AnimationExporter(figsize=(10, 10), dpi=100)

# Create animation for specific robot type
anim = exporter.animate_integrator(result, model,
    obstacles=obstacle_list,
    goal=goal_region,
    title="My Robot",
    fps=20
)

# Or for unicycle
anim = exporter.animate_unicycle(result, model,
    target=target_state,
    fps=20
)

# Or for manipulator
anim = exporter.animate_manipulator(result, model,
    target=target_config,
    fps=30
)

# Save
exporter.save_gif(anim, 'my_animation.gif')
```

### Comparison Animation

```python
from sim.animation import create_comparison_animation

# Multiple simulation results
results = [p_result, lqr_result, mpc_result]
labels = ['P-Control', 'LQR', 'MPC']

anim = create_comparison_animation(
    results, labels, model,
    robot_type='integrator',
    fps=20
)

# Save
exporter = AnimationExporter()
exporter.save_gif(anim, 'comparison.gif')
```

---

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
python -m pytest tests/test_models.py -v
python -m pytest tests/test_controllers.py -v
python -m pytest tests/test_mpc_rl.py -v
```

### Smoke Tests

Quick verification that the main components work:

```bash
# Test integrator
python run_smoke_test.py

# Test unicycle (after adding to smoke test)
python run_smoke_test_unicycle.py

# Test manipulator
python run_smoke_test_manipulator.py
```

---

## Examples

### Example 1: Complete Integrator Control Pipeline

```python
import numpy as np
from models.integrator import IntegratorModel
from controllers import LQRController
from sim.simulator import Simulator
from sim.animation import export_simulation

# Setup
model = IntegratorModel(tau=0.1)
ctrl = LQRController()
ctrl.design_for_model(model)
ctrl.set_target(np.zeros(2))

# Simulate
sim = Simulator(model, ctrl, disturbance_mode='random', seed=42)
result = sim.run(np.array([5.0, 5.0]), T=10.0)

# Analyze
stability = ctrl.analyze_stability()
metrics = result.compute_metrics()

print(f"Stable: {stability['is_stable']}")
print(f"Settling time: {metrics['settling_time']:.2f}s")
print(f"Final error: {result.error_norms[-1]:.4f}")

# Export animation
export_simulation(result, model, 'lqr_demo', format='gif')
```

### Example 2: Unicycle Path Following

```python
import numpy as np
from models.unicycle import UnicycleModel
from controllers.unicycle import PolarCoordinateController
from sim.simulator import Simulator

model = UnicycleModel(tau=0.1)
ctrl = PolarCoordinateController(k_rho=1.0, k_alpha=3.0, k_beta=-0.5)
ctrl.set_model(model)

# Go from (2,2,0) to (8,8,pi/2)
x0 = np.array([2.0, 2.0, 0.0])
target = np.array([8.0, 8.0, np.pi/2])
ctrl.set_target(target)

sim = Simulator(model, ctrl)
result = sim.run(x0, T=20.0, target=target)

# Check stability conditions
conditions = ctrl.verify_stability_conditions()
print(f"All stability conditions met: {conditions['all_satisfied']}")
```

### Example 3: Manipulator Trajectory Tracking

```python
import numpy as np
from models.manipulator import TwoLinkManipulator
from controllers.manipulator import ComputedTorqueController
from sim.simulator import Simulator

model = TwoLinkManipulator(tau=0.01)
ctrl = ComputedTorqueController(kp=100.0, kd=20.0)
ctrl.set_model(model)

# Move from hanging (0,0) to 45-45 degrees
x0 = np.array([0.0, 0.0, 0.0, 0.0])
target = np.array([np.pi/4, np.pi/4, 0.0, 0.0])
ctrl.set_target(target)

sim = Simulator(model, ctrl)
result = sim.run(x0, T=3.0, target=target)

# Visualize end-effector path
ee_positions = [model.forward_kinematics(s[:2])[1] for s in result.states]
```

### Example 4: Reach-Avoid with Obstacles

```python
import numpy as np
from models.integrator import IntegratorModel
from symbolic.grid_abstraction import GridAbstraction
from symbolic.reach_avoid import ReachAvoidPlanner, ReachAvoidController
from sim.simulator import Simulator

model = IntegratorModel(tau=0.1)

# Create abstraction
abstraction = GridAbstraction(model.x_bounds, resolution=(25, 25), model=model)

# Set goal
goal = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
abstraction.set_goal_region(goal)

# Add obstacles
obstacles = [
    np.array([[2, -5], [4, -5], [4, 5], [2, 5]]),
    np.array([[-4, 2], [-2, 2], [-2, 6], [-4, 6]])
]
abstraction.set_obstacles(obstacles)

# Plan and control
planner = ReachAvoidPlanner(abstraction)
planner.compute_value_function()
planner.compute_policy()

ctrl = ReachAvoidController(planner, waypoint_tolerance=0.5, kp=0.8)
ctrl.set_model(model)

sim = Simulator(model, ctrl)
result = sim.run(np.array([7.0, 7.0]), T=20.0)
```

### Example 5: Patrolling Behavior

```python
import numpy as np
from models.integrator import IntegratorModel
from symbolic.grid_abstraction import GridAbstraction
from symbolic.ltl_automata import RecurrenceController, create_patrol_regions
from sim.simulator import Simulator

model = IntegratorModel(tau=0.1)

# Create abstraction
abstraction = GridAbstraction(model.x_bounds, resolution=(20, 20), model=model)

# Create 4 patrol regions at corners
regions = create_patrol_regions(model.x_bounds, n_regions=4)

# Create patrol controller
ctrl = RecurrenceController(abstraction, regions, spec_type='patrol', kp=0.8)
ctrl.set_model(model)

# Compute patrol cycle
x0 = np.array([0.0, 0.0])
success = ctrl.compute_patrol_cycle(x0)

if success:
    sim = Simulator(model, ctrl)
    result = sim.run(x0, T=60.0)  # Long simulation for patrol

    info = ctrl.get_patrol_info()
    print(f"Patrolling {len(info['regions'])} regions")
    print(f"Cycle length: {info['cycle_length']} waypoints")
```

---

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the project root directory
   ```bash
   cd Robotics
   python your_script.py
   ```

2. **FFmpeg not found**: For MP4 export, install ffmpeg:
   - Windows: `choco install ffmpeg` or download from https://ffmpeg.org
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

3. **Slow MPC**: Reduce horizon or increase tau (sampling period)

4. **Manipulator diverges**: Use smaller tau (e.g., 0.01) for stiff dynamics

5. **RL not converging**: Increase episodes, adjust learning rate/sigma

---

## API Reference

For detailed API documentation, see docstrings in each module:

```python
# Get help on any class/function
help(IntegratorModel)
help(LQRController.design_for_model)
help(Simulator.run)
```

---

*Framework developed for discrete-time robot control education and research.*
