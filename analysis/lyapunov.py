"""
Lyapunov stability analysis.

Provides tools for constructing and verifying Lyapunov functions
for discrete-time systems.

Theory:
    A function V: R^n -> R is a Lyapunov function for x(t+1) = f(x(t))
    with equilibrium at x* if:

    1. V(x*) = 0
    2. V(x) > 0 for x != x*
    3. ΔV(x) = V(f(x)) - V(x) ≤ 0 for all x (stability)
       ΔV(x) = V(f(x)) - V(x) < 0 for x != x* (asymptotic stability)

For quadratic V(x) = x' P x with P > 0:
    - The system x(t+1) = A x(t) is stable if A' P A - P < 0
    - This is equivalent to eigenvalues of A inside unit circle

Limitations:
    - Analytic Lyapunov functions may not exist for all stable systems
    - Numerical verification is empirical, not a formal guarantee
    - Results depend on sampling of initial conditions and trajectories
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class LyapunovVerificationResult:
    """Result of Lyapunov verification."""
    is_valid: bool
    decrease_rate: float  # Average ΔV/V ratio
    max_increase: float   # Maximum ΔV observed (should be ≤ 0)
    violation_count: int  # Number of times ΔV > 0
    total_samples: int
    details: Dict[str, Any]


def verify_lyapunov_decrease(V: Callable[[np.ndarray], float],
                             f: Callable[[np.ndarray], np.ndarray],
                             initial_states: List[np.ndarray],
                             n_steps: int = 100,
                             tolerance: float = 1e-8) -> LyapunovVerificationResult:
    """
    Numerically verify Lyapunov function decrease along trajectories.

    This is an EMPIRICAL verification, not a formal proof. It checks
    that V decreases along simulated trajectories from given initial
    conditions. A passing result suggests stability but does not prove it.

    Args:
        V: Lyapunov function candidate V(x) -> float
        f: System dynamics f(x) -> x_next
        initial_states: List of initial states to test
        n_steps: Number of simulation steps per trajectory
        tolerance: Numerical tolerance for "zero"

    Returns:
        LyapunovVerificationResult with verification outcome
    """
    violations = 0
    total_samples = 0
    max_increase = -np.inf
    decrease_rates = []

    all_V_values = []
    all_delta_V = []

    for x0 in initial_states:
        x = np.asarray(x0, dtype=float)

        for _ in range(n_steps):
            V_current = V(x)

            if np.abs(V_current) < tolerance:
                # At equilibrium
                break

            x_next = f(x)
            V_next = V(x_next)

            delta_V = V_next - V_current

            all_V_values.append(V_current)
            all_delta_V.append(delta_V)

            if delta_V > tolerance:
                violations += 1

            max_increase = max(max_increase, delta_V)

            if V_current > tolerance:
                decrease_rates.append(delta_V / V_current)

            total_samples += 1
            x = x_next

    avg_decrease_rate = np.mean(decrease_rates) if decrease_rates else 0.0

    is_valid = violations == 0 and max_increase <= tolerance

    return LyapunovVerificationResult(
        is_valid=is_valid,
        decrease_rate=avg_decrease_rate,
        max_increase=max_increase,
        violation_count=violations,
        total_samples=total_samples,
        details={
            'V_values': np.array(all_V_values),
            'delta_V': np.array(all_delta_V),
            'decrease_rates': np.array(decrease_rates)
        }
    )


class LyapunovAnalyzer:
    """
    Lyapunov stability analyzer for discrete-time systems.

    Provides methods for:
    - Constructing quadratic Lyapunov functions
    - Verifying stability conditions
    - Numerical verification along trajectories
    """

    def __init__(self, model, controller):
        """
        Initialize analyzer.

        Args:
            model: Robot model
            controller: Controller with Lyapunov function
        """
        self.model = model
        self.controller = controller

    def construct_quadratic_lyapunov(self,
                                     target: np.ndarray,
                                     P: Optional[np.ndarray] = None) -> Callable:
        """
        Construct quadratic Lyapunov function V(x) = (x-target)' P (x-target).

        Args:
            target: Equilibrium state
            P: Positive definite matrix (default: identity)

        Returns:
            Lyapunov function V(x) -> float
        """
        target = np.asarray(target)

        if P is None:
            P = np.eye(len(target))

        def V(x):
            e = np.asarray(x) - target
            return float(e @ P @ e)

        return V

    def analyze_closed_loop_stability(self,
                                      target: np.ndarray,
                                      n_samples: int = 100,
                                      radius: float = 5.0,
                                      n_steps: int = 100) -> Dict[str, Any]:
        """
        Analyze closed-loop stability numerically.

        Simulates trajectories from random initial conditions and
        verifies Lyapunov decrease.

        Args:
            target: Target equilibrium
            n_samples: Number of random initial conditions
            radius: Sampling radius around target
            n_steps: Steps per trajectory

        Returns:
            Dictionary with analysis results
        """
        self.controller.set_target(target)

        # Sample initial conditions
        rng = np.random.default_rng(42)
        initial_states = []

        for _ in range(n_samples):
            # Random direction and distance
            direction = rng.standard_normal(self.model.n_states)
            direction = direction / np.linalg.norm(direction)
            distance = rng.uniform(0.1, radius)
            x0 = target + distance * direction

            # Clip to state bounds
            x0 = np.clip(x0, self.model.x_bounds[:, 0], self.model.x_bounds[:, 1])
            initial_states.append(x0)

        # Get Lyapunov function from controller if available
        if hasattr(self.controller, 'P') and self.controller.P is not None:
            P = self.controller.P
        else:
            P = np.eye(self.model.n_states)

        V = self.construct_quadratic_lyapunov(target, P)

        # Closed-loop dynamics
        def closed_loop(x):
            u, _ = self.controller.compute_control(x, 0.0)
            u = self.model.saturate_input(u)
            return self.model.dynamics(x, u, np.zeros(self.model.w_bounds.shape[0]))

        # Verify
        result = verify_lyapunov_decrease(V, closed_loop, initial_states, n_steps)

        return {
            'verification': result,
            'is_stable': result.is_valid,
            'average_decrease_rate': result.decrease_rate,
            'max_increase': result.max_increase,
            'n_violations': result.violation_count,
            'n_samples': n_samples,
            'n_steps': n_steps,
            'lyapunov_matrix': P,
            'note': 'EMPIRICAL verification only - not a formal stability proof'
        }

    def estimate_region_of_attraction(self,
                                      target: np.ndarray,
                                      n_directions: int = 36,
                                      max_radius: float = 10.0,
                                      n_steps: int = 200,
                                      convergence_tol: float = 0.1) -> Dict[str, Any]:
        """
        Estimate region of attraction numerically.

        For each direction from target, finds the maximum initial
        distance that leads to convergence.

        Args:
            target: Target equilibrium
            n_directions: Number of angular directions to test
            max_radius: Maximum radius to test
            n_steps: Simulation steps
            convergence_tol: Distance threshold for convergence

        Returns:
            Dictionary with ROA boundary estimates
        """
        self.controller.set_target(target)

        # Only works for 2D systems
        if self.model.n_states != 2:
            return {'error': 'ROA estimation only implemented for 2D systems'}

        angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        radii = []

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Binary search for maximum stable radius
            r_low, r_high = 0.0, max_radius

            for _ in range(10):  # 10 iterations of binary search
                r_test = (r_low + r_high) / 2
                x0 = target + r_test * direction

                # Clip to bounds
                x0 = np.clip(x0, self.model.x_bounds[:, 0], self.model.x_bounds[:, 1])

                # Simulate
                x = x0.copy()
                for _ in range(n_steps):
                    u, _ = self.controller.compute_control(x, 0.0)
                    u = self.model.saturate_input(u)
                    x = self.model.dynamics(x, u, np.zeros(2))

                # Check convergence
                if np.linalg.norm(x - target) < convergence_tol:
                    r_low = r_test
                else:
                    r_high = r_test

            radii.append(r_low)

        radii = np.array(radii)

        # Convert to boundary points
        boundary_x = target[0] + radii * np.cos(angles)
        boundary_y = target[1] + radii * np.sin(angles)

        return {
            'angles': angles,
            'radii': radii,
            'boundary': np.column_stack([boundary_x, boundary_y]),
            'min_radius': np.min(radii),
            'max_radius': np.max(radii),
            'mean_radius': np.mean(radii),
            'note': 'EMPIRICAL estimate - actual ROA may differ'
        }
