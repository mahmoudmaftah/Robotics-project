"""
Feedback Linearization Controller for Unicycle.

Transforms the nonlinear unicycle dynamics into a linear system
using input-output linearization.

The unicycle model:
    x1_dot = v * cos(theta)
    x2_dot = v * sin(theta)
    theta_dot = omega

Using a reference point (x_r, y_r) ahead of the robot at distance d:
    x_r = x + d * cos(theta)
    y_r = y + d * sin(theta)

The reference point dynamics become:
    x_r_dot = v * cos(theta) - d * omega * sin(theta)
    y_r_dot = v * sin(theta) + d * omega * cos(theta)

This can be written as: [x_r_dot; y_r_dot] = T(theta) * [v; omega]

where T(theta) = [cos(theta), -d*sin(theta)]
                 [sin(theta),  d*cos(theta)]

T is invertible for d != 0, so we can compute:
    [v; omega] = T^{-1}(theta) * [u1; u2]

where [u1; u2] is a new linear input.

Lyapunov Analysis:
    For the linearized system with proportional control u = -K*e,
    V(e) = e'e is a valid Lyapunov function with V_dot < 0.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class FeedbackLinearizationController(Controller):
    """
    Feedback linearization controller for unicycle.

    Uses input-output linearization with a reference point
    ahead of the robot to transform nonlinear dynamics to linear.
    """

    def __init__(self, d: float = 0.3, kp: float = 1.0,
                 name: str = "FeedbackLinearization"):
        """
        Initialize feedback linearization controller.

        Args:
            d: Distance to reference point (must be > 0)
            kp: Proportional gain for position tracking
            name: Controller name
        """
        super().__init__(name)
        if d <= 0:
            raise ValueError("Reference point distance d must be positive")
        self.d = d
        self.kp = kp

    def _get_reference_point(self, x: np.ndarray) -> np.ndarray:
        """
        Compute reference point position.

        Args:
            x: State [x, y, theta]

        Returns:
            Reference point [x_r, y_r]
        """
        return np.array([
            x[0] + self.d * np.cos(x[2]),
            x[1] + self.d * np.sin(x[2])
        ])

    def _get_transformation_matrix(self, theta: float) -> np.ndarray:
        """
        Get transformation matrix T(theta).

        Args:
            theta: Heading angle

        Returns:
            2x2 transformation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -self.d * s],
                         [s,  self.d * c]])

    def _get_inverse_transformation(self, theta: float) -> np.ndarray:
        """
        Get inverse transformation matrix T^{-1}(theta).

        det(T) = d * (cos^2 + sin^2) = d
        T^{-1} = (1/d) * [d*cos, d*sin; -sin, cos]

        Args:
            theta: Heading angle

        Returns:
            2x2 inverse transformation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s],
                         [-s/self.d, c/self.d]])

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute feedback linearization control.

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

        # Get reference point position
        ref_point = self._get_reference_point(x)

        # Target for reference point (same as target position)
        target_pos = self._target[:2]

        # Error in reference point coordinates
        error = ref_point - target_pos

        # Linear control law: auxiliary input
        u_aux = -self.kp * error

        # Transform back to original inputs
        T_inv = self._get_inverse_transformation(x[2])
        u = T_inv @ u_aux

        # Lyapunov function: squared distance of reference point to target
        V = 0.5 * np.dot(error, error)

        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error),
            'lyapunov': V,
            'reference_point': ref_point.copy(),
            'u_auxiliary': u_aux.copy(),
            'transformation_det': self.d
        }

        return u, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value.

        V(x) = 0.5 * ||ref_point - target||^2

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None:
            return None
        ref_point = self._get_reference_point(x)
        error = ref_point - self._target[:2]
        return 0.5 * float(np.dot(error, error))

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze stability of feedback linearized system.

        The linearized system is: e_dot = -kp * e
        Discrete: e(t+1) = (1 - tau*kp) * e(t)

        Stable if |1 - tau*kp| < 1, i.e., 0 < tau*kp < 2

        Returns:
            Stability analysis results
        """
        if self.model is None:
            return {'analyzed': False, 'reason': 'Model not set'}

        tau = self.model.tau
        eigenvalue = 1 - tau * self.kp

        return {
            'eigenvalue': eigenvalue,
            'is_stable': abs(eigenvalue) < 1,
            'stability_margin': 1 - abs(eigenvalue),
            'recommended_kp_range': (0, 2/tau),
            'note': 'Stability assumes perfect linearization (no disturbance)'
        }
