"""
Discrete-time simulation engine.

Runs closed-loop simulations with disturbance injection and
data logging for analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Attributes:
        time: Time vector
        states: State trajectory (T x n_states)
        inputs: Control inputs (T x n_inputs)
        disturbances: Applied disturbances (T x n_dist)
        diagnostics: List of controller diagnostics per step
        metadata: Additional simulation information
    """
    time: np.ndarray
    states: np.ndarray
    inputs: np.ndarray
    disturbances: np.ndarray
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lyapunov_values(self) -> np.ndarray:
        """Extract Lyapunov function values from diagnostics."""
        if not self.diagnostics:
            return np.array([])
        values = []
        for d in self.diagnostics:
            if 'lyapunov' in d:
                values.append(d['lyapunov'])
            else:
                values.append(np.nan)
        return np.array(values)

    @property
    def error_norms(self) -> np.ndarray:
        """Extract error norms from diagnostics."""
        if not self.diagnostics:
            return np.array([])
        return np.array([d.get('error_norm', np.nan) for d in self.diagnostics])

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics.

        Returns:
            Dictionary with settling time, overshoot, input energy, etc.
        """
        metrics = {}

        # Error convergence
        errors = self.error_norms
        if len(errors) > 0:
            final_error = errors[-1]
            metrics['final_error'] = final_error

            # Settling time (2% criterion)
            initial_error = errors[0]
            if initial_error > 0:
                threshold = 0.02 * initial_error
                settled_indices = np.where(errors < threshold)[0]
                if len(settled_indices) > 0:
                    metrics['settling_time'] = self.time[settled_indices[0]]
                else:
                    metrics['settling_time'] = np.inf

        # Input energy
        metrics['input_energy'] = float(np.sum(self.inputs**2) * (self.time[1] - self.time[0]))

        # Max input
        metrics['max_input'] = float(np.max(np.abs(self.inputs)))

        return metrics


class Simulator:
    """
    Discrete-time simulation engine.

    Runs closed-loop simulation of a robot model with a controller,
    optionally injecting bounded disturbances.
    """

    def __init__(self, model, controller,
                 disturbance_mode: str = 'random',
                 seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            model: RobotModel instance
            controller: Controller instance
            disturbance_mode: 'random', 'worst_case', 'none', or 'custom'
            seed: Random seed for reproducibility
        """
        self.model = model
        self.controller = controller
        self.disturbance_mode = disturbance_mode

        # Set up RNG
        self.rng = np.random.default_rng(seed)

        # Custom disturbance function
        self._custom_disturbance: Optional[Callable] = None

        # Link controller to model
        self.controller.set_model(model)

    def set_custom_disturbance(self, func: Callable[[np.ndarray, float], np.ndarray]) -> None:
        """
        Set custom disturbance function.

        Args:
            func: Function (state, time) -> disturbance
        """
        self._custom_disturbance = func
        self.disturbance_mode = 'custom'

    def _get_disturbance(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Generate disturbance based on mode.

        Args:
            x: Current state
            t: Current time

        Returns:
            w: Disturbance vector
        """
        if self.disturbance_mode == 'none':
            return np.zeros(self.model.w_bounds.shape[0])

        elif self.disturbance_mode == 'random':
            return self.model.sample_disturbance(self.rng)

        elif self.disturbance_mode == 'worst_case':
            # For worst-case: disturbance opposes control direction
            # This is a simplification; true worst-case depends on dynamics
            return self.rng.choice([-1, 1], size=self.model.w_bounds.shape[0]) * \
                   self.model.w_bounds[:, 1]

        elif self.disturbance_mode == 'custom' and self._custom_disturbance is not None:
            return self._custom_disturbance(x, t)

        else:
            return np.zeros(self.model.w_bounds.shape[0])

    def run(self, x0: np.ndarray, T: float,
            target: Optional[np.ndarray] = None) -> SimulationResult:
        """
        Run closed-loop simulation.

        Args:
            x0: Initial state
            T: Total simulation time (seconds)
            target: Goal state (optional, uses controller's target if not provided)

        Returns:
            SimulationResult with full trajectory data
        """
        x0 = np.asarray(x0, dtype=float)

        if target is not None:
            self.controller.set_target(target)

        # Reset controller state
        self.controller.reset()

        # Time vector
        n_steps = int(T / self.model.tau) + 1
        time = np.linspace(0, T, n_steps)

        # Pre-allocate arrays
        states = np.zeros((n_steps, self.model.n_states))
        inputs = np.zeros((n_steps, self.model.n_inputs))
        disturbances = np.zeros((n_steps, self.model.w_bounds.shape[0]))
        diagnostics = []

        # Initial state
        states[0] = x0

        # Simulation loop
        for k in range(n_steps - 1):
            x = states[k]
            t = time[k]

            # Compute control
            u_raw, diag = self.controller.compute_control(x, t)

            # Saturate input
            u = self.model.saturate_input(u_raw)

            # Get disturbance
            w = self._get_disturbance(x, t)

            # Store data
            inputs[k] = u
            disturbances[k] = w
            diag['u_raw'] = u_raw.copy()
            diag['u_saturated'] = u.copy()
            diag['saturated'] = not np.allclose(u_raw, u)
            diagnostics.append(diag)

            # Propagate dynamics
            states[k + 1] = self.model.dynamics(x, u, w)

        # Final step diagnostics
        u_raw, diag = self.controller.compute_control(states[-1], time[-1])
        u = self.model.saturate_input(u_raw)
        inputs[-1] = u
        disturbances[-1] = self._get_disturbance(states[-1], time[-1])
        diagnostics.append(diag)

        # Metadata
        metadata = {
            'model': self.model.__class__.__name__,
            'controller': self.controller.name,
            'tau': self.model.tau,
            'disturbance_mode': self.disturbance_mode,
            'x0': x0.copy(),
            'target': self.controller.target.copy() if self.controller.target is not None else None
        }

        return SimulationResult(
            time=time,
            states=states,
            inputs=inputs,
            disturbances=disturbances,
            diagnostics=diagnostics,
            metadata=metadata
        )

    def run_batch(self, initial_states: List[np.ndarray], T: float,
                  target: Optional[np.ndarray] = None) -> List[SimulationResult]:
        """
        Run multiple simulations from different initial conditions.

        Args:
            initial_states: List of initial state vectors
            T: Simulation time
            target: Goal state

        Returns:
            List of SimulationResult objects
        """
        results = []
        for x0 in initial_states:
            results.append(self.run(x0, T, target))
        return results
