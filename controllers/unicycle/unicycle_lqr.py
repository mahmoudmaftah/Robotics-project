"""
LQR Controller for Unicycle (Linearized).

Since the unicycle is nonlinear, LQR is applied to a linearized model
around a nominal trajectory or operating point.

Linearization around (x_eq, u_eq):
    The discrete unicycle:
        x1(t+1) = x1(t) + tau * v * cos(theta)
        x2(t+1) = x2(t) + tau * v * sin(theta)
        theta(t+1) = theta(t) + tau * omega

    Linearized around theta = theta_eq, v = v_eq:
        A = I + tau * [0, 0, -v_eq*sin(theta_eq)]
                      [0, 0,  v_eq*cos(theta_eq)]
                      [0, 0,  0]

        B = tau * [cos(theta_eq), 0]
                  [sin(theta_eq), 0]
                  [0,             1]

For path following, we track a reference trajectory and apply LQR
to the error dynamics.

Lyapunov function:
    V(e) = e' P e where P solves DARE for linearized system.

Limitations:
    - Only locally valid near linearization point
    - Performance degrades far from operating point
    - May need gain scheduling for large deviations
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class UnicycleLQRController(Controller):
    """
    LQR controller for unicycle using local linearization.

    Linearizes around the current heading and applies LQR
    for trajectory tracking.
    """

    def __init__(self, Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 v_nom: float = 0.5,
                 name: str = "UnicycleLQR"):
        """
        Initialize unicycle LQR controller.

        Args:
            Q: State cost matrix (3x3)
            R: Input cost matrix (2x2)
            v_nom: Nominal forward velocity for linearization
            name: Controller name
        """
        super().__init__(name)

        # Default cost matrices
        self._Q = Q if Q is not None else np.diag([10.0, 10.0, 1.0])
        self._R = R if R is not None else np.diag([1.0, 1.0])

        self.v_nom = v_nom
        self._K = None
        self._P = None
        self._last_theta_lin = None

    def _linearize(self, theta: float, v: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize unicycle dynamics around given state.

        Args:
            theta: Heading angle for linearization
            v: Forward velocity for linearization
            tau: Sampling period

        Returns:
            A, B: Linearized system matrices
        """
        c, s = np.cos(theta), np.sin(theta)

        A = np.eye(3)
        A[0, 2] = -tau * v * s
        A[1, 2] = tau * v * c

        B = tau * np.array([[c, 0],
                            [s, 0],
                            [0, 1]])

        return A, B

    def _design_lqr(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design LQR gain for linearized system.

        Args:
            A: State matrix
            B: Input matrix

        Returns:
            K: LQR gain
            P: DARE solution
        """
        try:
            P = linalg.solve_discrete_are(A, B, self._Q, self._R)
            K = np.linalg.solve(self._R + B.T @ P @ B, B.T @ P @ A)
            return K, P
        except Exception:
            # Fallback to simple proportional gain
            K = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
            P = self._Q
            return K, P

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute LQR control with online linearization.

        Args:
            x: Current state [x, y, theta]
            t: Current time

        Returns:
            u: Control input [v, omega]
            diagnostics: Controller info
        """
        x = np.asarray(x, dtype=float)

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        if self.model is None:
            tau = 0.1
        else:
            tau = self.model.tau

        # Compute error
        error = x - self._target

        # Wrap angle error
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))

        # Linearize around current heading
        theta_lin = x[2]

        # Re-design LQR if heading changed significantly
        if (self._K is None or self._last_theta_lin is None or
            abs(theta_lin - self._last_theta_lin) > 0.3):

            A, B = self._linearize(theta_lin, self.v_nom, tau)
            self._K, self._P = self._design_lqr(A, B)
            self._last_theta_lin = theta_lin

        # LQR control: u = -K @ error
        u_delta = -self._K @ error

        # Add nominal forward velocity toward target
        dist_to_target = np.linalg.norm(x[:2] - self._target[:2])
        angle_to_target = np.arctan2(self._target[1] - x[1],
                                      self._target[0] - x[0])
        heading_error = np.arctan2(np.sin(angle_to_target - x[2]),
                                   np.cos(angle_to_target - x[2]))

        # Modulate velocity based on heading alignment
        v_nom_adjusted = self.v_nom * np.cos(heading_error) * min(1.0, dist_to_target)

        u = np.array([v_nom_adjusted + u_delta[0],
                      u_delta[1]])

        # Lyapunov function
        V = float(error @ self._P @ error) if self._P is not None else np.linalg.norm(error)**2

        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error[:2]),
            'lyapunov': V,
            'gain_matrix': self._K.copy() if self._K is not None else None,
            'linearization_point': theta_lin,
            'heading_error': heading_error,
            'dist_to_target': dist_to_target
        }

        return u, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value.

        V(x) = (x - target)' P (x - target)

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None or self._P is None:
            return None

        error = np.asarray(x) - self._target
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        return float(error @ self._P @ error)

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze stability of LQR controller.

        Returns:
            Stability analysis (local result)
        """
        if self._K is None or self.model is None:
            return {'analyzed': False, 'reason': 'Controller not initialized'}

        tau = self.model.tau
        A, B = self._linearize(self._last_theta_lin or 0, self.v_nom, tau)

        # Closed-loop system
        A_cl = A - B @ self._K

        eigvals = np.linalg.eigvals(A_cl)
        spectral_radius = np.max(np.abs(eigvals))

        return {
            'eigenvalues': eigvals,
            'spectral_radius': spectral_radius,
            'is_locally_stable': spectral_radius < 1.0,
            'linearization_point': self._last_theta_lin,
            'limitations': [
                'Only locally valid near linearization point',
                'Requires gain scheduling for global stability',
                'Performance degrades far from nominal trajectory'
            ]
        }
