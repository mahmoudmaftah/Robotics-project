"""
Backstepping Controller for Two-Link Manipulator.

Backstepping is a recursive Lyapunov-based design technique for
nonlinear systems in strict-feedback form.

For the manipulator, we can write:
    q_dot = v  (virtual input)
    M(q)v_dot + C(q,v)v + g(q) = tau

Step 1: Position subsystem
    Define z1 = q - q_d (position error)
    Design virtual control: alpha = q_dot_d - k1*z1
    Lyapunov: V1 = 0.5 * z1' * z1
    V1_dot = z1' * (q_dot - q_dot_d) = z1' * (v - q_dot_d)

    With v = alpha: V1_dot = -k1 * z1' * z1 < 0

Step 2: Velocity subsystem
    Define z2 = v - alpha = q_dot - q_dot_d + k1*z1
    Design tau to make z2 -> 0

    Lyapunov: V = V1 + 0.5 * z2' * M(q) * z2

    V_dot = -k1*z1'*z1 + z1'*z2 + z2'*M*z2_dot + 0.5*z2'*M_dot*z2

    Using skew-symmetry of M_dot - 2C and the manipulator dynamics:

    tau = M(q)(alpha_dot - k2*z2) + C(q,q_dot)q_dot + g(q) - z1

    This gives: V_dot = -k1*z1'*z1 - k2*z2'*M*z2 < 0

The backstepping approach provides a constructive Lyapunov function
and systematic gain design.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class BacksteppingController(Controller):
    """
    Backstepping controller for two-link manipulator.

    Provides systematic Lyapunov-based design with guaranteed stability.
    """

    def __init__(self, k1: float = 5.0, k2: float = 10.0,
                 name: str = "Backstepping"):
        """
        Initialize backstepping controller.

        Args:
            k1: Position error gain (determines convergence rate of z1)
            k2: Velocity error gain (determines convergence rate of z2)
            name: Controller name
        """
        super().__init__(name)
        self.k1 = k1
        self.k2 = k2

        # Desired trajectory
        self._q_d = None
        self._q_dot_d = None
        self._q_ddot_d = None

        # Previous values for derivative computation
        self._prev_alpha = None
        self._prev_t = None

    def reset(self) -> None:
        """Reset controller state."""
        self._prev_alpha = None
        self._prev_t = None

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
            self._q_ddot_d = np.zeros(2)
        elif len(target) == 4:
            self._target = target.copy()
            self._q_d = target[:2]
            self._q_dot_d = target[2:]
            self._q_ddot_d = np.zeros(2)
        else:
            raise ValueError("Target must be 2D or 4D")

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute backstepping control.

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

        # Get time step
        tau_dt = self.model.tau

        # Current state
        q = x[:2]
        q_dot = x[2:]

        # Step 1: Position error
        z1 = q - self._q_d
        # Wrap angles
        z1[0] = np.arctan2(np.sin(z1[0]), np.cos(z1[0]))
        z1[1] = np.arctan2(np.sin(z1[1]), np.cos(z1[1]))

        # Virtual control (stabilizing function)
        alpha = self._q_dot_d - self.k1 * z1

        # Step 2: Velocity error
        z2 = q_dot - alpha

        # Compute alpha_dot (derivative of virtual control)
        if self._prev_alpha is not None and self._prev_t is not None:
            dt = t - self._prev_t if t > self._prev_t else tau_dt
            alpha_dot = (alpha - self._prev_alpha) / max(dt, 1e-6)
        else:
            # First call: use desired acceleration
            alpha_dot = self._q_ddot_d - self.k1 * (q_dot - self._q_dot_d)

        # Update stored values
        self._prev_alpha = alpha.copy()
        self._prev_t = t

        # Get model matrices
        M = self.model.mass_matrix(q)
        c = self.model.coriolis_vector(q, q_dot)
        g = self.model.gravity_vector(q)

        # Backstepping control law
        # tau = M*(alpha_dot - k2*z2) + C*q_dot + g - z1
        tau = M @ (alpha_dot - self.k2 * z2) + c + g - z1

        # Lyapunov function
        # V = 0.5*z1'*z1 + 0.5*z2'*M*z2
        V = 0.5 * (z1 @ z1 + z2 @ M @ z2)

        diagnostics = {
            'error': np.concatenate([z1, z2]),
            'error_norm': np.linalg.norm(z1),
            'velocity_error_norm': np.linalg.norm(z2),
            'lyapunov': V,
            'z1': z1.copy(),
            'z2': z2.copy(),
            'alpha': alpha.copy(),
            'alpha_dot': alpha_dot.copy()
        }

        return tau, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function.

        V = 0.5*z1'*z1 + 0.5*z2'*M(q)*z2

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None or self.model is None:
            return None

        q = x[:2]
        q_dot = x[2:]

        z1 = q - self._q_d
        z1[0] = np.arctan2(np.sin(z1[0]), np.cos(z1[0]))
        z1[1] = np.arctan2(np.sin(z1[1]), np.cos(z1[1]))

        alpha = self._q_dot_d - self.k1 * z1
        z2 = q_dot - alpha

        M = self.model.mass_matrix(q)
        return 0.5 * float(z1 @ z1 + z2 @ M @ z2)

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze stability of backstepping controller.

        The Lyapunov function V = 0.5*(z1'*z1 + z2'*M*z2) satisfies:
        V_dot = -k1*z1'*z1 - k2*z2'*M*z2 < 0 for (z1,z2) != 0

        Returns:
            Stability analysis
        """
        return {
            'lyapunov_function': 'V = 0.5*(z1\'*z1 + z2\'*M(q)*z2)',
            'lyapunov_derivative': 'V_dot = -k1*||z1||^2 - k2*z2\'*M*z2',
            'stability_type': 'globally_asymptotically_stable',
            'convergence_rates': {
                'position': f'z1 converges with rate k1 = {self.k1}',
                'velocity': f'z2 converges with rate ~k2 = {self.k2}'
            },
            'gains': {'k1': self.k1, 'k2': self.k2},
            'robustness': 'Sensitive to model uncertainty; consider adaptive extensions',
            'note': 'Constructive Lyapunov design with guaranteed GAS'
        }
