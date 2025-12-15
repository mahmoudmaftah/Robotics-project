"""
PD Controller with Gravity Compensation for Two-Link Manipulator.

A simpler alternative to computed torque that compensates only for
gravity while using PD feedback for tracking.

Control law:
    tau = Kp*(q_d - q) + Kd*(q_dot_d - q_dot) + g(q)

The gravity term g(q) compensates for the static load, while PD
feedback handles tracking.

Lyapunov Analysis:
    V = 0.5 * q_dot' * M(q) * q_dot + 0.5 * e' * Kp * e

    V_dot = q_dot' * M * q_ddot + 0.5 * q_dot' * M_dot * q_dot - e' * Kp * q_dot

    Using M*q_ddot = tau - c - g and the skew-symmetry property of M_dot - 2C:

    V_dot = q_dot' * (Kp*e + Kd*e_dot + g - c - g) - e' * Kp * q_dot
          = q_dot' * Kp * e + q_dot' * Kd * e_dot - q_dot' * c - e' * Kp * q_dot

    At regulation (q_dot_d = 0, e_dot = -q_dot):
    V_dot = -q_dot' * Kd * q_dot <= 0

    Asymptotic stability follows from LaSalle.

Advantages over computed torque:
    - Simpler implementation (no M, C computation in real-time)
    - More robust to model uncertainty in M, C
    - Only gravity model needed

Disadvantages:
    - Slower convergence
    - Coupling effects not compensated
    - Non-zero tracking error for fast trajectories
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class PDGravityCompensation(Controller):
    """
    PD controller with gravity compensation for manipulator.

    Compensates static gravity load while using PD feedback for tracking.
    Simpler and more robust than computed torque.
    """

    def __init__(self, Kp: Optional[np.ndarray] = None,
                 Kd: Optional[np.ndarray] = None,
                 name: str = "PD+Gravity"):
        """
        Initialize PD + gravity compensation controller.

        Args:
            Kp: Position gain matrix (2x2, default: 50*I)
            Kd: Velocity gain matrix (2x2, default: 10*I)
            name: Controller name
        """
        super().__init__(name)

        # Default gains (lower than computed torque since no M compensation)
        self._Kp = Kp if Kp is not None else 50.0 * np.eye(2)
        self._Kd = Kd if Kd is not None else 10.0 * np.eye(2)

        # Desired state
        self._q_d = None
        self._q_dot_d = None

    def set_target(self, target: np.ndarray) -> None:
        """
        Set target joint configuration.

        Args:
            target: Target [theta1, theta2] or [theta1, theta2, dtheta1, dtheta2]
        """
        target = np.asarray(target, dtype=float)

        if len(target) == 2:
            self._target = np.array([target[0], target[1], 0.0, 0.0])
            self._q_d = target[:2]
            self._q_dot_d = np.zeros(2)
        elif len(target) == 4:
            self._target = target.copy()
            self._q_d = target[:2]
            self._q_dot_d = target[2:]
        else:
            raise ValueError("Target must be 2D or 4D")

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute PD + gravity compensation control.

        tau = Kp*e + Kd*e_dot + g(q)

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

        # Current state
        q = x[:2]
        q_dot = x[2:]

        # Errors
        e = self._q_d - q
        e_dot = self._q_dot_d - q_dot

        # Wrap angle errors
        e[0] = np.arctan2(np.sin(e[0]), np.cos(e[0]))
        e[1] = np.arctan2(np.sin(e[1]), np.cos(e[1]))

        # Gravity compensation
        g = self.model.gravity_vector(q)

        # PD control + gravity
        tau = self._Kp @ e + self._Kd @ e_dot + g

        # Lyapunov function
        M = self.model.mass_matrix(q)
        V = 0.5 * (q_dot @ M @ q_dot + e @ self._Kp @ e)

        diagnostics = {
            'error': np.concatenate([e, e_dot]),
            'error_norm': np.linalg.norm(e),
            'velocity_error_norm': np.linalg.norm(e_dot),
            'lyapunov': V,
            'position_error': e.copy(),
            'velocity_error': e_dot.copy(),
            'gravity_compensation': g.copy(),
            'pd_torque': (self._Kp @ e + self._Kd @ e_dot).copy()
        }

        return tau, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function.

        V = 0.5 * q_dot' * M(q) * q_dot + 0.5 * e' * Kp * e

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None or self.model is None:
            return None

        q = x[:2]
        q_dot = x[2:]
        e = self._q_d - q

        e[0] = np.arctan2(np.sin(e[0]), np.cos(e[0]))
        e[1] = np.arctan2(np.sin(e[1]), np.cos(e[1]))

        M = self.model.mass_matrix(q)
        return 0.5 * float(q_dot @ M @ q_dot + e @ self._Kp @ e)

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze stability properties.

        Returns:
            Stability analysis
        """
        # Check positive definiteness of gains
        kp_eigvals = np.linalg.eigvalsh(self._Kp)
        kd_eigvals = np.linalg.eigvalsh(self._Kd)

        kp_positive = np.all(kp_eigvals > 0)
        kd_positive = np.all(kd_eigvals > 0)

        return {
            'Kp_eigenvalues': kp_eigvals,
            'Kd_eigenvalues': kd_eigvals,
            'Kp_positive_definite': kp_positive,
            'Kd_positive_definite': kd_positive,
            'is_stable': kp_positive and kd_positive,
            'stability_type': 'asymptotically_stable' if (kp_positive and kd_positive) else 'unstable',
            'lyapunov_certificate': 'V = 0.5*(q_dot\'*M*q_dot + e\'*Kp*e)',
            'note': 'Global asymptotic stability for regulation (q_dot_d = 0)'
        }
