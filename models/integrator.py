"""
Discrete-time integrator model.

Model 1 from Symbolic_control_lecture-7.pdf:
    x1(t+1) = x1(t) + τ (u1(t) + w1(t))
    x2(t+1) = x2(t) + τ (u2(t) + w2(t))

State constraints: X = [-10,10] × [-10,10]
Input constraints: U = [-1,1] × [-1,1]
Disturbance set: W = [-0.05, 0.05] × [-0.05, 0.05]
"""

import numpy as np
from typing import Optional, Tuple
from .base import RobotModel


class IntegratorModel(RobotModel):
    """
    2D discrete-time integrator robot model.

    This is the simplest model: a point mass in 2D that moves with
    velocity directly controlled by input, plus additive disturbance.

    Dynamics:
        x(t+1) = x(t) + τ * (u(t) + w(t))

    The system is linear, fully controllable, and stable under
    simple proportional or LQR control.
    """

    def __init__(self, tau: float = 0.1):
        """
        Initialize integrator model.

        Args:
            tau: Sampling period (default 0.1s)
        """
        self.n_states = 2
        self.n_inputs = 2
        super().__init__(tau)

    def _setup_constraints(self) -> None:
        """Set up state, input, and disturbance constraints per lecture spec."""
        # State constraints: X = [-10, 10] × [-10, 10]
        self.x_bounds = np.array([[-10.0, 10.0],
                                   [-10.0, 10.0]])

        # Input constraints: U = [-1, 1] × [-1, 1]
        self.u_bounds = np.array([[-1.0, 1.0],
                                   [-1.0, 1.0]])

        # Disturbance set: W = [-0.05, 0.05] × [-0.05, 0.05]
        self.w_bounds = np.array([[-0.05, 0.05],
                                   [-0.05, 0.05]])

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute next state.

        x(t+1) = x(t) + τ * (u(t) + w(t))

        Args:
            x: Current state [x1, x2]
            u: Control input [u1, u2]
            w: Disturbance [w1, w2] (defaults to zero)

        Returns:
            x_next: Next state
        """
        if w is None:
            w = np.zeros(2)

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        w = np.asarray(w, dtype=float)

        x_next = x + self.tau * (u + w)
        return x_next

    def get_linearization(self, x_eq: np.ndarray,
                          u_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get linearized system matrices.

        For the integrator, the system is already linear:
            x(t+1) = I @ x(t) + τ*I @ u(t)

        So A = I, B = τ*I (ignoring disturbance for control design).

        Args:
            x_eq: Equilibrium state (not used for linear system)
            u_eq: Equilibrium input (not used for linear system)

        Returns:
            A: State matrix (2x2 identity)
            B: Input matrix (τ * 2x2 identity)
        """
        A = np.eye(2)
        B = self.tau * np.eye(2)
        return A, B

    def get_continuous_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get continuous-time system matrices for LQR design.

        Continuous dynamics: ẋ = u
        So A_c = 0, B_c = I

        Returns:
            A_c: Continuous state matrix (2x2 zeros)
            B_c: Continuous input matrix (2x2 identity)
        """
        A_c = np.zeros((2, 2))
        B_c = np.eye(2)
        return A_c, B_c
