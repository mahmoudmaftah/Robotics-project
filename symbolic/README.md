# Symbolic Control Synthesis

A Python implementation of **Symbolic Control** for nonlinear dynamical systems. This project provides tools for abstracting continuous systems into finite-state models and synthesizing controllers that are **correct by construction**.

## 📹 Demo

https://github.com/user-attachments/assets/bb76446c-4d5f-4a19-ab58-dbe8937e7217


---

## 🎯 Overview

Traditional control methods often struggle with complex specifications (safety constraints, temporal logic) and nonlinear dynamics. **Symbolic Control** addresses this by:

1. **Abstracting** a continuous system into a discrete (finite) model
2. **Synthesizing** a controller on the discrete model
3. **Concretizing** the controller back to the physical system

This approach provides **formal guarantees** of correctness and handles rich specifications like safety, reachability, and temporal logic.

---

## ✨ Features

- **Multiple Dynamics Models:**
  - 2D Integrator (simple motion with noise)
  - 3D Unicycle (non-holonomic robot with heading)
  - 4D Robotic Manipulator (two-link planar arm)

- **Synthesis Capabilities:**
  - Safety synthesis (invariance)
  - Reachability synthesis
  - Combined safety + reachability
  - Regular language specifications via NFA/Regex

- **Interactive GUI:**
  - Draw obstacles and target regions
  - Configure discretization parameters
  - Real-time synthesis progress
  - Trajectory simulation and visualization

- **Visualization Tools:**
  - Grid abstraction visualization
  - Post() (successor) cell visualization
  - Controller and winning set display

---

## 📁 Project Structure

```
symbolic/
├── README.md                    # This file
├── gui.py                       # Full GUI with regex/automata specifications
├── gui_simple.py                # Simplified GUI for safety + reachability
├── visualize_post.py            # Interactive Post() visualization tool
├── symbolic_control/            # Core library
│   ├── __init__.py
│   ├── dynamics.py              # Dynamics models (Integrator, Unicycle, Manipulator)
│   ├── abstraction.py           # Grid discretization and transitions
│   ├── synthesis.py             # Safety and reachability synthesis
│   ├── product_synthesis.py     # Product automaton for complex specs
│   └── nfa.py                   # NFA and regex-to-automaton conversion
├── docs/
│   ├── explanation.md           # Theory and methodology explanation
│   └── subapproximation.md      # Sub-approximation details
└── images/                      # Generated visualizations
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- tqdm

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/symbolic-control.git
   cd symbolic-control
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib tqdm
   ```

### Running the GUI

**Simple Synthesis GUI** (recommended for beginners):
```bash
python gui_simple.py
```

**Full GUI** with regex specifications:
```bash
python gui.py
```

**Interactive Post() Visualizer**:
```bash
python visualize_post.py
```

---

## 📖 Usage

### GUI Workflow

1. **Select Dynamics Model**: Choose between Integrator, Unicycle, or Manipulator
2. **Configure Parameters**:
   - State bounds (workspace limits)
   - Discretization η (grid cell size)
   - Control discretization
   - Sampling period τ
3. **Define Regions**:
   - Draw obstacles (red regions to avoid)
   - Draw target region (green region to reach)
4. **Run Synthesis**: Click "Run Synthesis" to compute the controller
5. **Simulate**: Click on the grid to set a start position and run simulation

### Programmatic Usage

```python
from symbolic_control import (
    IntegratorDynamics, 
    Abstraction, 
    Synthesis
)
import numpy as np

# 1. Create dynamics
dynamics = IntegratorDynamics(
    tau=0.4,           # Sampling period
    w_bound=0.01,      # Disturbance bound
    u_values=np.linspace(-1, 1, 5)
)

# 2. Create abstraction
state_bounds = np.array([[0, 10], [0, 10]])  # 2D workspace
abstraction = Abstraction(
    dynamics=dynamics,
    state_bounds=state_bounds,
    eta=0.5  # Grid cell size
)

# 3. Compute transitions
abstraction.compute_transitions()

# 4. Define regions and run synthesis
# ... (see examples in gui_simple.py)
```

---

## 🧮 Theory

The symbolic control methodology consists of three steps:

### 1. Symbolic Abstraction (Continuous → Discrete)
- Partition state space into a uniform grid
- Compute transitions using over-approximation (Post operator)
- Each grid cell becomes a symbolic state

### 2. Controller Synthesis (Discrete Domain)
- **Safety**: Find maximal controlled invariant set
- **Reachability**: Compute winning set via backward iteration
- **Complex Specs**: Use product automaton with NFA

### 3. Concretization (Discrete → Continuous)
- Map symbolic controller back to continuous system
- Acts as a lookup table: state → cell → control




## 📚 References

- Girard, A., Meyer, P.-J., & Saoud, A. (2024). *Approches symboliques pour le contrôle des systèmes non linéaires*
- Tabuada, P. (2009). *Verification and Control of Hybrid Systems: A Symbolic Approach*


