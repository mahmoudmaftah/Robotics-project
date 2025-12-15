"""
Sliding Mode Controller (SMC) for Unicycle.

Sliding mode control provides robust tracking despite disturbances
and model uncertainties by forcing the system state onto a sliding
surface and keeping it there.

Sliding surface design:
    For position tracking, define sliding surfaces:
    s1 = e_x + lambda * integral(e_x)  (for x tracking)
    s2 = e_theta + lambda * integral(e_theta)  (for heading)

    Or simply: s = e + lambda * e_dot

Control law:
    u = u_eq + u_sw
    where u_eq is equivalent control (nominal)
    and u_sw = -K * sign(s) is switching control

To reduce chattering, we use saturation function instead of sign:
    sat(s/phi) = s/phi if |s| < phi, else sign(s)

Lyapunov function:
    V = 0.5 * s' * s
    V_dot = s' * s_dot = s' * (e_dot + lambda * e)

With proper control, V_dot < 0 when s != 0.

Note: SMC for unicycle is challenging due to underactuation.
This implementation uses a hierarchical approach: outer loop for
position, inner loop for heading alignment.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class SlidingModeController(Controller):
    """
    Sliding mode controller for unicycle trajectory tracking.

    Uses hierarchical sliding surfaces for position and heading control.
    Includes boundary layer to reduce chattering.
    """

    def __init__(self, k_pos: float = 2.0, k_theta: float = 3.0,
                 lambda_pos: float = 1.0, lambda_theta: float = 1.5,
                 phi_pos: float = 0.3, phi_theta: float = 0.2,
                 name: str = "SlidingMode"):
        """
        Initialize sliding mode controller.

        Args:
            k_pos: Position sliding gain
            k_theta: Heading sliding gain
            lambda_pos: Position sliding surface slope
            lambda_theta: Heading sliding surface slope
            phi_pos: Position boundary layer thickness
            phi_theta: Heading boundary layer thickness
            name: Controller name
        """
        super().__init__(name)
        self.k_pos = k_pos
        self.k_theta = k_theta
        self.lambda_pos = lambda_pos
        self.lambda_theta = lambda_theta
        self.phi_pos = phi_pos
        self.phi_theta = phi_theta

        # State for integral terms
        self._integral_error = np.zeros(2)
        self._prev_error = None
        self._dt = 0.1

    def reset(self) -> None:
        """Reset controller state."""
        self._integral_error = np.zeros(2)
        self._prev_error = None

    def _sat(self, s: float, phi: float) -> float:
        """
        Saturation function (smooth approximation of sign).

        Args:
            s: Sliding variable
            phi: Boundary layer thickness

        Returns:
            Saturated value in [-1, 1]
        """
        if abs(s) < phi:
            return s / phi
        else:
            return np.sign(s)

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute sliding mode control.

        Hierarchical approach:
        1. Compute desired heading to face target
        2. Heading control (omega) to align with desired heading
        3. Forward velocity (v) when roughly aligned

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

        # Get time step from model if available
        if self.model is not None:
            self._dt = self.model.tau

        # Position error
        e_pos = np.array([self._target[0] - x[0],
                          self._target[1] - x[1]])
        e_pos_norm = np.linalg.norm(e_pos)

        # Desired heading to target
        if e_pos_norm > 0.01:
            theta_desired = np.arctan2(e_pos[1], e_pos[0])
        else:
            theta_desired = x[2]  # Keep current heading at target

        # Heading error
        e_theta = self._wrap_angle(theta_desired - x[2])

        # Update integral error (for sliding surface)
        self._integral_error += e_pos * self._dt
        self._integral_error = np.clip(self._integral_error, -5, 5)  # Anti-windup

        # Compute error derivative (approximation)
        if self._prev_error is not None:
            e_dot = (e_pos - self._prev_error) / self._dt
        else:
            e_dot = np.zeros(2)
        self._prev_error = e_pos.copy()

        # Sliding surfaces
        # Position sliding surface (scalar: distance-based)
        s_pos = e_pos_norm + self.lambda_pos * np.linalg.norm(self._integral_error[:2])

        # Heading sliding surface
        s_theta = e_theta

        # Control computation
        # Heading control (omega) - always active
        omega_eq = self.lambda_theta * e_theta  # Equivalent control
        omega_sw = self.k_theta * self._sat(s_theta, self.phi_theta)
        omega = omega_eq + omega_sw

        # Forward velocity - active when heading is roughly correct
        heading_aligned = abs(e_theta) < np.pi / 4  # Within 45 degrees

        if heading_aligned and e_pos_norm > 0.05:
            v_eq = self.lambda_pos * e_pos_norm * np.cos(e_theta)
            v_sw = self.k_pos * self._sat(s_pos, self.phi_pos) * np.cos(e_theta)
            v = v_eq + v_sw
        else:
            # Turn in place or at target
            v = 0.1 * e_pos_norm * np.cos(e_theta)  # Slow approach

        # Ensure v is positive (unicycle constraint)
        v = max(0.0, v)

        u = np.array([v, omega])

        # Lyapunov function: V = 0.5 * (s_pos^2 + s_theta^2)
        V = 0.5 * (s_pos**2 + s_theta**2)

        diagnostics = {
            'error': e_pos.copy(),
            'error_norm': e_pos_norm,
            'lyapunov': V,
            'sliding_surface_pos': s_pos,
            'sliding_surface_theta': s_theta,
            'theta_error': e_theta,
            'heading_aligned': heading_aligned,
            'in_boundary_layer': abs(s_theta) < self.phi_theta
        }

        return u, diagnostics

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value.

        V = 0.5 * (s_pos^2 + s_theta^2)

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None:
            return None

        e_pos = np.array([self._target[0] - x[0], self._target[1] - x[1]])
        e_pos_norm = np.linalg.norm(e_pos)

        if e_pos_norm > 0.01:
            theta_desired = np.arctan2(e_pos[1], e_pos[0])
        else:
            theta_desired = x[2]

        e_theta = self._wrap_angle(theta_desired - x[2])

        s_pos = e_pos_norm
        s_theta = e_theta

        return 0.5 * (s_pos**2 + s_theta**2)

    def analyze_robustness(self) -> Dict[str, Any]:
        """
        Analyze robustness properties of SMC.

        Returns:
            Robustness analysis
        """
        return {
            'disturbance_rejection': 'Matched disturbances rejected when |d| < k',
            'chattering_mitigation': f'Boundary layers: phi_pos={self.phi_pos}, phi_theta={self.phi_theta}',
            'reaching_time_bound': f'Finite reaching time with gains k_pos={self.k_pos}, k_theta={self.k_theta}',
            'limitations': [
                'Chattering may occur at high gains',
                'Boundary layer introduces steady-state error',
                'Underactuated system requires hierarchical design'
            ],
            'note': 'SMC provides robustness but formal guarantees depend on disturbance bounds'
        }
