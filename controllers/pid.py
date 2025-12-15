"""
PID Controller for discrete-time systems.

Implements discrete PID with anti-windup:
    u(t) = Kp*e(t) + Ki*Σe(τ)*dt + Kd*(e(t)-e(t-1))/dt

where e(t) = target - x(t).
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base import Controller


class PIDController(Controller):
    """
    Discrete-time PID controller with anti-windup.

    Supports multi-dimensional state spaces with diagonal gains.
    """

    def __init__(self,
                 Kp: float = 1.0,
                 Ki: float = 0.0,
                 Kd: float = 0.0,
                 dt: float = 0.1,
                 integral_limit: float = 10.0,
                 name: str = "PID"):
        """
        Initialize PID controller.

        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            dt: Time step for discrete integration
            integral_limit: Anti-windup limit for integral term
            name: Controller name
        """
        super().__init__(name)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_limit = integral_limit

        # Internal state
        self._integral = None
        self._prev_error = None

    def reset(self) -> None:
        """Reset integral and derivative state."""
        self._integral = None
        self._prev_error = None

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute PID control action.

        Args:
            x: Current state
            t: Current time

        Returns:
            u: Control input
            diagnostics: Contains P, I, D terms and error info
        """
        x = np.asarray(x, dtype=float)

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        # Compute error (note: PID typically uses target - x)
        error = self._target - x

        # Initialize integral and previous error
        if self._integral is None:
            self._integral = np.zeros_like(error)
        if self._prev_error is None:
            self._prev_error = error.copy()

        # Proportional term
        P_term = self.Kp * error

        # Integral term with anti-windup
        self._integral += error * self.dt
        self._integral = np.clip(self._integral,
                                  -self.integral_limit,
                                  self.integral_limit)
        I_term = self.Ki * self._integral

        # Derivative term
        if self.dt > 0:
            D_term = self.Kd * (error - self._prev_error) / self.dt
        else:
            D_term = np.zeros_like(error)

        # Total control
        u = P_term + I_term + D_term

        # Update previous error
        self._prev_error = error.copy()

        # Lyapunov-like function: quadratic error
        V = 0.5 * np.dot(error, error)

        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error),
            'P_term': P_term.copy(),
            'I_term': I_term.copy(),
            'D_term': D_term.copy(),
            'integral': self._integral.copy(),
            'lyapunov': V
        }

        return u, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov-like function.

        V(x) = 0.5 * ||target - x||^2

        Note: This is not a true Lyapunov function for PID
        (especially with integral term), but serves as an
        error metric.

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None:
            return None
        error = self._target - np.asarray(x)
        return 0.5 * float(np.dot(error, error))
