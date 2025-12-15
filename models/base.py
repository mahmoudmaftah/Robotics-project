"""
Base class for discrete-time robot models.

Provides a standard interface for all robot models used in control design
and simulation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class RobotModel(ABC):
    """
    Abstract base class for discrete-time robot models.

    All models implement the discrete-time dynamics:
        x(t+1) = f(x(t), u(t), w(t))

    where:
        x: state vector
        u: control input
        w: disturbance (bounded)

    Attributes:
        tau: Sampling period (seconds)
        n_states: Dimension of state space
        n_inputs: Dimension of control input
        x_bounds: State space bounds (n_states x 2 array)
        u_bounds: Input bounds (n_inputs x 2 array)
        w_bounds: Disturbance bounds (n_states x 2 array or similar)
    """

    def __init__(self, tau: float = 0.1):
        """
        Initialize robot model.

        Args:
            tau: Sampling period in seconds
        """
        self.tau = tau
        self._setup_constraints()

    @abstractmethod
    def _setup_constraints(self) -> None:
        """Set up state, input, and disturbance constraints."""
        pass

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute next state given current state, input, and disturbance.

        Args:
            x: Current state vector
            u: Control input vector
            w: Disturbance vector (optional, defaults to zero)

        Returns:
            x_next: Next state vector
        """
        pass

    def saturate_input(self, u: np.ndarray) -> np.ndarray:
        """
        Saturate control input to respect input constraints.

        Args:
            u: Raw control input

        Returns:
            u_sat: Saturated control input within bounds
        """
        return np.clip(u, self.u_bounds[:, 0], self.u_bounds[:, 1])

    def sample_disturbance(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample a random disturbance within bounds.

        Args:
            rng: Random number generator (for reproducibility)

        Returns:
            w: Disturbance sample
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.w_bounds[:, 0], self.w_bounds[:, 1])

    def is_state_valid(self, x: np.ndarray) -> bool:
        """
        Check if state is within valid bounds.

        Args:
            x: State vector

        Returns:
            True if state is within bounds
        """
        return np.all(x >= self.x_bounds[:, 0]) and np.all(x <= self.x_bounds[:, 1])

    def get_linearization(self, x_eq: np.ndarray,
                          u_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get linearized system matrices A, B around equilibrium.

        For discrete-time system: x(t+1) ≈ A @ (x - x_eq) + B @ (u - u_eq) + x_eq

        Default implementation uses numerical differentiation.
        Override for analytical linearization.

        Args:
            x_eq: Equilibrium state
            u_eq: Equilibrium input

        Returns:
            A: State matrix (n_states x n_states)
            B: Input matrix (n_states x n_inputs)
        """
        eps = 1e-6
        A = np.zeros((self.n_states, self.n_states))
        B = np.zeros((self.n_states, self.n_inputs))

        f0 = self.dynamics(x_eq, u_eq, np.zeros(self.w_bounds.shape[0]))

        # Compute A matrix (df/dx)
        for i in range(self.n_states):
            x_plus = x_eq.copy()
            x_plus[i] += eps
            f_plus = self.dynamics(x_plus, u_eq, np.zeros(self.w_bounds.shape[0]))
            A[:, i] = (f_plus - f0) / eps

        # Compute B matrix (df/du)
        for i in range(self.n_inputs):
            u_plus = u_eq.copy()
            u_plus[i] += eps
            f_plus = self.dynamics(x_eq, u_plus, np.zeros(self.w_bounds.shape[0]))
            B[:, i] = (f_plus - f0) / eps

        return A, B

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tau={self.tau})"
