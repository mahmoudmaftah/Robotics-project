"""
Discrete-time two-link planar manipulator model.

Model 3 from Symbolic_control_lecture-7.pdf:
    x(t+1) = x(t) + τ * f(x(t), u(t))

where x = [θ₁, θ₂, θ̇₁, θ̇₂] and:
    f(x,u) = [θ̇; M(θ)⁻¹(u - c(θ,θ̇) - g(θ))]

Parameters: m1=m2=1.0 kg, ℓ1=ℓ2=0.5 m, g=9.81 m/s²
"""

import numpy as np
from typing import Optional, Tuple
from .base import RobotModel


class TwoLinkManipulator(RobotModel):
    """
    Discrete-time two-link planar manipulator.

    State: x = [θ₁, θ₂, θ̇₁, θ̇₂]
        θ₁, θ₂: Joint angles
        θ̇₁, θ̇₂: Joint velocities

    Input: u = [τ₁, τ₂] (joint torques)

    Uses Euler discretization of the continuous-time Lagrangian dynamics:
        M(θ)θ̈ + C(θ,θ̇)θ̇ + g(θ) = u
    """

    def __init__(self, tau: float = 0.01,
                 m1: float = 1.0, m2: float = 1.0,
                 l1: float = 0.5, l2: float = 0.5,
                 g: float = 9.81):
        """
        Initialize two-link manipulator.

        Args:
            tau: Sampling period (default 0.01s - smaller for stiff dynamics)
            m1, m2: Link masses (kg)
            l1, l2: Link lengths (m)
            g: Gravitational acceleration (m/s²)
        """
        self.n_states = 4
        self.n_inputs = 2

        # Physical parameters
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g_acc = g

        # Link center of mass (at midpoint)
        self.lc1 = l1 / 2
        self.lc2 = l2 / 2

        # Moments of inertia (thin rod about center)
        self.I1 = m1 * l1**2 / 12
        self.I2 = m2 * l2**2 / 12

        super().__init__(tau)

    def _setup_constraints(self) -> None:
        """Set up state, input, and disturbance constraints."""
        # State bounds: angles in [-π, π], velocities reasonable
        self.x_bounds = np.array([[-np.pi, np.pi],
                                   [-np.pi, np.pi],
                                   [-10.0, 10.0],
                                   [-10.0, 10.0]])

        # Torque limits (reasonable for small manipulator)
        self.u_bounds = np.array([[-10.0, 10.0],
                                   [-10.0, 10.0]])

        # Small disturbance torques
        self.w_bounds = np.array([[-0.1, 0.1],
                                   [-0.1, 0.1],
                                   [-0.1, 0.1],
                                   [-0.1, 0.1]])

    def mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute mass/inertia matrix M(θ).

        Args:
            theta: Joint angles [θ₁, θ₂]

        Returns:
            M: 2x2 mass matrix
        """
        m1, m2 = self.m1, self.m2
        l1, lc1, lc2 = self.l1, self.lc1, self.lc2
        I1, I2 = self.I1, self.I2

        c2 = np.cos(theta[1])

        M11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2) + I1 + I2
        M12 = m2*(lc2**2 + l1*lc2*c2) + I2
        M22 = m2*lc2**2 + I2

        return np.array([[M11, M12],
                         [M12, M22]])

    def coriolis_vector(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal term c(θ, θ̇).

        Args:
            theta: Joint angles
            theta_dot: Joint velocities

        Returns:
            c: 2-vector of Coriolis terms
        """
        m2 = self.m2
        l1, lc2 = self.l1, self.lc2

        s2 = np.sin(theta[1])
        h = m2 * l1 * lc2 * s2

        c1 = -h * theta_dot[1]**2 - 2*h * theta_dot[0] * theta_dot[1]
        c2 = h * theta_dot[0]**2

        return np.array([c1, c2])

    def gravity_vector(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gravity term g(θ).

        Args:
            theta: Joint angles

        Returns:
            g: 2-vector of gravity torques
        """
        m1, m2, g = self.m1, self.m2, self.g_acc
        l1, lc1, lc2 = self.l1, self.lc1, self.lc2

        g1 = (m1*lc1 + m2*l1) * g * np.cos(theta[0]) + \
             m2 * lc2 * g * np.cos(theta[0] + theta[1])
        g2 = m2 * lc2 * g * np.cos(theta[0] + theta[1])

        return np.array([g1, g2])

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute next state using Euler integration.

        Args:
            x: Current state [θ₁, θ₂, θ̇₁, θ̇₂]
            u: Control input [τ₁, τ₂]
            w: Disturbance (added to state derivative)

        Returns:
            x_next: Next state
        """
        if w is None:
            w = np.zeros(4)

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        w = np.asarray(w, dtype=float)

        theta = x[:2]
        theta_dot = x[2:]

        # Compute dynamics: M(θ)θ̈ = u - c(θ,θ̇) - g(θ)
        M = self.mass_matrix(theta)
        c = self.coriolis_vector(theta, theta_dot)
        g = self.gravity_vector(theta)

        # Solve for acceleration
        theta_ddot = np.linalg.solve(M, u - c - g)

        # Euler integration
        x_next = np.zeros(4)
        x_next[:2] = theta + self.tau * theta_dot + self.tau * w[:2]
        x_next[2:] = theta_dot + self.tau * theta_ddot + self.tau * w[2:]

        # Wrap angles to [-π, π]
        x_next[0] = np.arctan2(np.sin(x_next[0]), np.cos(x_next[0]))
        x_next[1] = np.arctan2(np.sin(x_next[1]), np.cos(x_next[1]))

        return x_next

    def forward_kinematics(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector position.

        Args:
            theta: Joint angles [θ₁, θ₂]

        Returns:
            p1: Position of first joint (elbow)
            p2: Position of end-effector
        """
        p1 = np.array([self.l1 * np.cos(theta[0]),
                       self.l1 * np.sin(theta[0])])

        p2 = p1 + np.array([self.l2 * np.cos(theta[0] + theta[1]),
                            self.l2 * np.sin(theta[0] + theta[1])])

        return p1, p2
