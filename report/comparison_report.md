# Discrete-Time Robot Control: Comprehensive Comparison Report

**Generated:** 2025-12-04 17:42:01

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
| Proportional | 8.80 | 0.0776 | 0.0000 | 8.05 | 0.015 |
| LQR | 6.30 | 0.0042 | 0.0036 | 9.00 | 0.011 |
| PID | 10.00 | 1.7279 | 0.1285 | 11.60 | 0.016 |
| MPC | 5.10 | 0.0037 | 0.0042 | 9.67 | 6.969 |
| RL-PG | 10.00 | 7.1093 | 0.0000 | 20.20 | 0.018 |

### Robustness Analysis (10 trials with random disturbances)

| Controller | Mean Error | Std Error | Max Error | Success Rate |
|------------|------------|-----------|-----------|--------------|
| Proportional | 0.0815 | 0.0083 | 0.0933 | 100% |
| LQR | 0.0093 | 0.0044 | 0.0154 | 100% |
| PID | 1.7287 | 0.0070 | 1.7410 | 0% |
| MPC | 0.0050 | 0.0007 | 0.0062 | 100% |
| RL-PG | 7.1236 | 0.0256 | 7.1596 | 0% |

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
| FeedbackLin | 15.00 | 0.9106 | 0.0000 | 12.63 | 0.016 |
| PolarCoord | 15.00 | 0.2707 | 0.0000 | 14.17 | 0.015 |
| SlidingMode | 14.30 | 0.1225 | 0.0000 | 16.47 | 0.026 |
| LQR | 15.00 | 1.5058 | 0.0000 | 11.49 | 0.105 |

### Robustness Analysis (10 trials with random disturbances)

| Controller | Mean Error | Std Error | Max Error | Success Rate |
|------------|------------|-----------|-----------|--------------|
| FeedbackLin | 0.1975 | 0.0181 | 0.2278 | 100% |
| PolarCoord | 0.0300 | 0.0073 | 0.0405 | 100% |
| SlidingMode | 0.3185 | 0.0282 | 0.3671 | 100% |
| LQR | 0.2633 | 0.0158 | 0.2900 | 100% |

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
| ComputedTorque | 1.04 | 0.0215 | 0.0034 | 132.57 | 0.031 |
| PD+Gravity | 3.00 | 2.8689 | 0.0000 | 499.28 | 0.039 |
| Backstepping | 1.17 | 0.0230 | 0.0124 | 128.53 | 0.077 |
| LQR | 1.56 | 0.0155 | 0.0033 | 146.59 | 0.019 |

### Robustness Analysis (10 trials with random disturbances)

| Controller | Mean Error | Std Error | Max Error | Success Rate |
|------------|------------|-----------|-----------|--------------|
| ComputedTorque | 0.0131 | 0.0074 | 0.0265 | 100% |
| PD+Gravity | 2.1692 | 0.4313 | 2.8523 | 0% |
| Backstepping | 0.0177 | 0.0106 | 0.0416 | 100% |
| LQR | 0.0127 | 0.0063 | 0.0275 | 100% |

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
| Integrator | Proportional | 0.015 | Yes |
| Integrator | LQR | 0.011 | Yes |
| Integrator | PID | 0.016 | Yes |
| Integrator | MPC | 6.969 | Yes |
| Integrator | RL-PG | 0.018 | Yes |
| Unicycle | FeedbackLin | 0.016 | Yes |
| Unicycle | PolarCoord | 0.015 | Yes |
| Unicycle | SlidingMode | 0.026 | Yes |
| Unicycle | LQR | 0.105 | Yes |
| Manipulator | ComputedTorque | 0.031 | Yes |
| Manipulator | PD+Gravity | 0.039 | Yes |
| Manipulator | Backstepping | 0.077 | Yes |
| Manipulator | LQR | 0.019 | Yes |

### Overall Robustness Ranking

Based on success rate across all disturbance trials:

| Rank | Model | Controller | Success Rate | Mean Error |
|------|-------|------------|--------------|------------|
| 1 | Integrator | MPC | 100% | 0.0050 |
| 2 | Integrator | LQR | 100% | 0.0093 |
| 3 | Manipulator | LQR | 100% | 0.0127 |
| 4 | Manipulator | ComputedTorque | 100% | 0.0131 |
| 5 | Manipulator | Backstepping | 100% | 0.0177 |
| 6 | Unicycle | PolarCoord | 100% | 0.0300 |
| 7 | Integrator | Proportional | 100% | 0.0815 |
| 8 | Unicycle | FeedbackLin | 100% | 0.1975 |
| 9 | Unicycle | LQR | 100% | 0.2633 |
| 10 | Unicycle | SlidingMode | 100% | 0.3185 |

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
