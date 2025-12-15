"""
Discrete-time unicycle model.

Model 2 from Symbolic_control_lecture-7.pdf:
    x1(t+1) = x1(t) + τ (u1(t) * cos(x3(t)) + w1(t))
    x2(t+1) = x2(t) + τ (u1(t) * sin(x3(t)) + w2(t))
    x3(t+1) = x3(t) + τ (u2(t) + w3(t))   (mod 2π)

State constraints: X = [0,10] × [0,10] × [-π,π]
Input constraints: U = [0.25,1] × [-1,1]
Disturbance set: W = [-0.05, 0.05]³
"""

import numpy as np
from typing import Optional, Tuple
from .base import RobotModel


class UnicycleModel(RobotModel):
    """
    Discrete-time unicycle (nonholonomic mobile robot) model.

    State: [x, y, θ] where (x,y) is position and θ is heading angle.
    Input: [v, ω] where v is forward velocity and ω is angular velocity.

    This is a nonlinear system due to the cos/sin terms coupling
    velocity to heading angle.
    """

    def __init__(self, tau: float = 0.1):
        """
        Initialize unicycle model.

        Args:
            tau: Sampling period (default 0.1s)
        """
        self.n_states = 3
        self.n_inputs = 2
        super().__init__(tau)

    def _setup_constraints(self) -> None:
        """Set up constraints per lecture specification."""
        # State constraints: X = [0,10] × [0,10] × [-π,π]
        self.x_bounds = np.array([[0.0, 10.0],
                                   [0.0, 10.0],
                                   [-np.pi, np.pi]])

        # Input constraints: U = [0.25,1] × [-1,1]
        # Note: velocity must be positive (forward motion only)
        self.u_bounds = np.array([[0.25, 1.0],
                                   [-1.0, 1.0]])

        # Disturbance set: W = [-0.05, 0.05]³
        self.w_bounds = np.array([[-0.05, 0.05],
                                   [-0.05, 0.05],
                                   [-0.05, 0.05]])

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute next state.

        Args:
            x: Current state [x1, x2, θ]
            u: Control input [v, ω]
            w: Disturbance [w1, w2, w3] (defaults to zero)

        Returns:
            x_next: Next state
        """
        if w is None:
            w = np.zeros(3)

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        w = np.asarray(w, dtype=float)

        x_next = np.zeros(3)
        x_next[0] = x[0] + self.tau * (u[0] * np.cos(x[2]) + w[0])
        x_next[1] = x[1] + self.tau * (u[0] * np.sin(x[2]) + w[1])
        x_next[2] = x[2] + self.tau * (u[1] + w[2])

        # Wrap angle to [-π, π]
        x_next[2] = np.arctan2(np.sin(x_next[2]), np.cos(x_next[2]))

        return x_next

    def get_linearization(self, x_eq: np.ndarray,
                          u_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get linearized system matrices around equilibrium.

        The unicycle is nonlinear, so linearization depends on the
        operating point. For a stationary robot at heading θ_eq
        with velocity v_eq:

        A = I + τ * [0, 0, -v*sin(θ)]
                    [0, 0,  v*cos(θ)]
                    [0, 0,  0       ]

        B = τ * [cos(θ), 0]
                [sin(θ), 0]
                [0,      1]

        Args:
            x_eq: Equilibrium state
            u_eq: Equilibrium input

        Returns:
            A, B: Linearized system matrices
        """
        theta = x_eq[2]
        v = u_eq[0]

        A = np.eye(3)
        A[0, 2] = -self.tau * v * np.sin(theta)
        A[1, 2] = self.tau * v * np.cos(theta)

        B = self.tau * np.array([[np.cos(theta), 0],
                                  [np.sin(theta), 0],
                                  [0, 1]])

        return A, B
