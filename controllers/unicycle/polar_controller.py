"""
Polar Coordinate Controller for Unicycle.

Uses polar coordinates (rho, alpha, beta) for point stabilization.
This approach provides globally asymptotically stable regulation
to a target pose.

Coordinate transformation:
    rho = sqrt((x_t - x)^2 + (y_t - y)^2)   # distance to target
    alpha = atan2(y_t - y, x_t - x) - theta  # angle to target
    beta = alpha + theta - theta_t           # orientation error

Control law (Astolfi, 1999):
    v = k_rho * rho * cos(alpha)
    omega = k_alpha * alpha + k_rho * sin(alpha)*cos(alpha)/alpha * (alpha + k_beta*beta)

For alpha = 0, use L'Hopital: sin(alpha)*cos(alpha)/alpha -> 1

Lyapunov function:
    V(rho, alpha, beta) = 0.5 * (rho^2 + alpha^2 + k_beta * beta^2)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class PolarCoordinateController(Controller):
    """
    Polar coordinate controller for unicycle stabilization.

    Provides globally asymptotically stable regulation to target pose.
    Based on Astolfi (1999) exponential stabilization approach.
    """

    def __init__(self, k_rho: float = 1.0, k_alpha: float = 3.0,
                 k_beta: float = -1.0, name: str = "PolarController"):
        """
        Initialize polar coordinate controller.

        Args:
            k_rho: Distance gain (> 0)
            k_alpha: Heading to target gain (> 0)
            k_beta: Final orientation gain (< 0 for stability)
            name: Controller name

        Note: Stability requires k_rho > 0, k_alpha > 0, k_beta < 0
              and k_alpha - k_rho > 0
        """
        super().__init__(name)
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta

        # Target orientation (default: facing target at arrival)
        self._theta_target = 0.0

    def set_target(self, target: np.ndarray) -> None:
        """
        Set target pose.

        Args:
            target: Target [x, y, theta] or [x, y] (theta defaults to 0)
        """
        target = np.asarray(target, dtype=float)
        if len(target) == 2:
            self._target = np.array([target[0], target[1], 0.0])
        else:
            self._target = target.copy()
        self._theta_target = self._target[2] if len(self._target) > 2 else 0.0

    def _compute_polar_coords(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Transform Cartesian state to polar error coordinates.

        Args:
            x: State [x, y, theta]

        Returns:
            (rho, alpha, beta): Polar error coordinates
        """
        dx = self._target[0] - x[0]
        dy = self._target[1] - x[1]

        rho = np.sqrt(dx**2 + dy**2)

        # Angle to target from robot
        angle_to_target = np.arctan2(dy, dx)

        # Alpha: heading error (how much to turn to face target)
        alpha = self._wrap_angle(angle_to_target - x[2])

        # Beta: final orientation error
        beta = self._wrap_angle(alpha + x[2] - self._theta_target)

        return rho, alpha, beta

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute polar coordinate control.

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

        # Convert to polar coordinates
        rho, alpha, beta = self._compute_polar_coords(x)

        # Near target: switch to pure rotation if needed
        rho_threshold = 0.05
        if rho < rho_threshold:
            # At target position, just correct orientation
            v = 0.0
            theta_error = self._wrap_angle(self._theta_target - x[2])
            omega = self.k_alpha * theta_error
        else:
            # Control law
            v = self.k_rho * rho * np.cos(alpha)

            # Handle alpha near zero (L'Hopital limit)
            if abs(alpha) < 1e-6:
                sinc_term = 1.0
            else:
                sinc_term = np.sin(alpha) * np.cos(alpha) / alpha

            omega = (self.k_alpha * alpha +
                     self.k_rho * sinc_term * (alpha + self.k_beta * beta))

        u = np.array([v, omega])

        # Lyapunov function
        V = 0.5 * (rho**2 + alpha**2 + abs(self.k_beta) * beta**2)

        # Position error for comparison
        pos_error = np.sqrt((self._target[0] - x[0])**2 +
                            (self._target[1] - x[1])**2)

        diagnostics = {
            'error': np.array([self._target[0] - x[0],
                               self._target[1] - x[1]]),
            'error_norm': pos_error,
            'lyapunov': V,
            'rho': rho,
            'alpha': alpha,
            'beta': beta,
            'polar_coords': np.array([rho, alpha, beta])
        }

        return u, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value.

        V = 0.5 * (rho^2 + alpha^2 + |k_beta| * beta^2)

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None:
            return None
        rho, alpha, beta = self._compute_polar_coords(x)
        return 0.5 * (rho**2 + alpha**2 + abs(self.k_beta) * beta**2)

    def verify_stability_conditions(self) -> Dict[str, Any]:
        """
        Verify stability conditions for the control gains.

        Stability requires:
            k_rho > 0
            k_alpha > 0
            k_beta < 0
            k_alpha - k_rho > 0

        Returns:
            Verification results
        """
        conditions = {
            'k_rho_positive': self.k_rho > 0,
            'k_alpha_positive': self.k_alpha > 0,
            'k_beta_negative': self.k_beta < 0,
            'k_alpha_minus_k_rho_positive': (self.k_alpha - self.k_rho) > 0
        }

        all_satisfied = all(conditions.values())

        return {
            'conditions': conditions,
            'all_satisfied': all_satisfied,
            'gains': {'k_rho': self.k_rho, 'k_alpha': self.k_alpha, 'k_beta': self.k_beta},
            'note': 'Global asymptotic stability proven for continuous-time system'
        }
