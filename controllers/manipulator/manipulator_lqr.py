"""
LQR Controller for Two-Link Manipulator (Linearized).

Applies LQR to the manipulator linearized around an equilibrium point.

Linearization of manipulator dynamics around (q_eq, u_eq):
    M(q_eq)q_ddot + C(q_eq, 0)q_dot + g(q_eq) = u_eq

At equilibrium with gravity compensation: u_eq = g(q_eq)

Linearized continuous dynamics:
    M(q_eq) * delta_q_ddot + dC/dq_dot|_eq * delta_q_dot + dg/dq|_eq * delta_q = delta_u

In state-space form with x = [delta_q; delta_q_dot]:
    x_dot = A*x + B*delta_u

where:
    A = [0, I; -M^{-1}*G, 0]  (G = dg/dq at equilibrium)
    B = [0; M^{-1}]

For discrete-time, we discretize using matrix exponential or Euler.

LQR minimizes: J = sum(x'Qx + u'Ru)

Lyapunov function: V(x) = x'Px where P solves DARE

Limitations:
    - Only locally valid near linearization point
    - Gravity compensation needed as feedforward
    - Performance degrades for large deviations
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class ManipulatorLQRController(Controller):
    """
    LQR controller for linearized manipulator dynamics.

    Uses gravity compensation as feedforward with LQR feedback.
    """

    def __init__(self, Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 name: str = "ManipulatorLQR"):
        """
        Initialize manipulator LQR controller.

        Args:
            Q: State cost matrix (4x4, default: diag([100,100,10,10]))
            R: Input cost matrix (2x2, default: I)
            name: Controller name
        """
        super().__init__(name)

        # Default cost matrices
        self._Q = Q if Q is not None else np.diag([100.0, 100.0, 10.0, 10.0])
        self._R = R if R is not None else np.eye(2)

        self._K = None
        self._P = None
        self._q_eq = None  # Equilibrium (target) position

    def set_target(self, target: np.ndarray) -> None:
        """
        Set target configuration and design LQR.

        Args:
            target: Target [theta1, theta2] or [theta1, theta2, 0, 0]
        """
        target = np.asarray(target, dtype=float)

        if len(target) == 2:
            self._target = np.array([target[0], target[1], 0.0, 0.0])
        else:
            self._target = target.copy()

        self._q_eq = self._target[:2]

        # Design LQR if model is set
        if self.model is not None:
            self._design_lqr()

    def set_model(self, model) -> None:
        """Set model and design LQR."""
        self.model = model
        if self._q_eq is not None:
            self._design_lqr()

    def _compute_gravity_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of gravity vector dg/dq numerically.

        Args:
            q: Joint configuration

        Returns:
            G: 2x2 Jacobian matrix
        """
        eps = 1e-6
        G = np.zeros((2, 2))
        g0 = self.model.gravity_vector(q)

        for i in range(2):
            q_plus = q.copy()
            q_plus[i] += eps
            g_plus = self.model.gravity_vector(q_plus)
            G[:, i] = (g_plus - g0) / eps

        return G

    def _design_lqr(self) -> None:
        """Design LQR controller for linearized system."""
        if self.model is None or self._q_eq is None:
            return

        tau = self.model.tau

        # Get model matrices at equilibrium
        M = self.model.mass_matrix(self._q_eq)
        M_inv = np.linalg.inv(M)
        G = self._compute_gravity_jacobian(self._q_eq)

        # Continuous-time state-space matrices
        # x = [q - q_eq; q_dot], x_dot = A_c*x + B_c*u
        A_c = np.zeros((4, 4))
        A_c[0:2, 2:4] = np.eye(2)
        A_c[2:4, 0:2] = -M_inv @ G

        B_c = np.zeros((4, 2))
        B_c[2:4, :] = M_inv

        # Discretize using Euler (simple) or matrix exponential
        # Simple Euler discretization for now
        A_d = np.eye(4) + tau * A_c
        B_d = tau * B_c

        # Solve discrete algebraic Riccati equation
        try:
            self._P = linalg.solve_discrete_are(A_d, B_d, self._Q, self._R)
            self._K = np.linalg.solve(self._R + B_d.T @ self._P @ B_d,
                                      B_d.T @ self._P @ A_d)
        except Exception:
            # Fallback: simple gains
            self._K = np.array([[50, 0, 10, 0],
                               [0, 50, 0, 10]])
            self._P = self._Q

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute LQR control with gravity feedforward.

        tau = -K*(x - x_eq) + g(q)

        Args:
            x: Current state [theta1, theta2, theta1_dot, theta2_dot]
            t: Current time

        Returns:
            u: Control torques [tau1, tau2]
            diagnostics: Controller info
        """
        x = np.asarray(x, dtype=float)

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        if self._K is None:
            self._design_lqr()

        # Compute error state
        error = x - self._target

        # Wrap angle errors
        error[0] = np.arctan2(np.sin(error[0]), np.cos(error[0]))
        error[1] = np.arctan2(np.sin(error[1]), np.cos(error[1]))

        # LQR feedback
        tau_fb = -self._K @ error

        # Gravity feedforward (at current position)
        g = self.model.gravity_vector(x[:2])

        # Total control
        tau = tau_fb + g

        # Lyapunov function
        V = float(error @ self._P @ error) if self._P is not None else np.linalg.norm(error)**2

        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error[:2]),
            'velocity_error_norm': np.linalg.norm(error[2:]),
            'lyapunov': V,
            'feedback_torque': tau_fb.copy(),
            'gravity_feedforward': g.copy(),
            'gain_matrix': self._K.copy() if self._K is not None else None
        }

        return tau, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function.

        V(x) = (x - x_eq)' P (x - x_eq)

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None or self._P is None:
            return None

        error = np.asarray(x) - self._target
        error[0] = np.arctan2(np.sin(error[0]), np.cos(error[0]))
        error[1] = np.arctan2(np.sin(error[1]), np.cos(error[1]))

        return float(error @ self._P @ error)

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze closed-loop stability.

        Returns:
            Stability analysis
        """
        if self._K is None or self.model is None:
            return {'analyzed': False, 'reason': 'Controller not designed'}

        tau = self.model.tau
        M = self.model.mass_matrix(self._q_eq)
        M_inv = np.linalg.inv(M)
        G = self._compute_gravity_jacobian(self._q_eq)

        # Reconstruct discretized system
        A_c = np.zeros((4, 4))
        A_c[0:2, 2:4] = np.eye(2)
        A_c[2:4, 0:2] = -M_inv @ G

        B_c = np.zeros((4, 2))
        B_c[2:4, :] = M_inv

        A_d = np.eye(4) + tau * A_c
        B_d = tau * B_c

        # Closed-loop system
        A_cl = A_d - B_d @ self._K

        eigvals = np.linalg.eigvals(A_cl)
        spectral_radius = np.max(np.abs(eigvals))

        return {
            'eigenvalues': eigvals,
            'spectral_radius': spectral_radius,
            'is_locally_stable': spectral_radius < 1.0,
            'equilibrium_point': self._q_eq.tolist() if self._q_eq is not None else None,
            'limitations': [
                'Only locally valid near equilibrium',
                'Requires gravity compensation feedforward',
                'Nonlinear effects not captured far from operating point'
            ]
        }
