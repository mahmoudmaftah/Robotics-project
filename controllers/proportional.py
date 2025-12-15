"""
Proportional (P) Controller.

Simple linear controller: u = -K @ (x - x_target)

Lyapunov analysis for integrator model:
    V(e) = e' @ P @ e  (quadratic in error e = x - x_target)

For the discrete integrator x(t+1) = x(t) + τ*u(t):
    With u = -K @ e, we get e(t+1) = (I - τK) @ e(t)

    The system is stable if eigenvalues of (I - τK) are inside unit circle.
    For K = k*I with k > 0: eigenvalues are (1 - τk).
    Stable if |1 - τk| < 1, i.e., 0 < τk < 2.

    Choosing τk < 1 (underdamped) gives guaranteed convergence.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base import Controller


class ProportionalController(Controller):
    """
    Proportional position controller.

    Implements u = -K @ (x - x_target) with optional gain matrix K.
    Provides quadratic Lyapunov function for stability analysis.
    """

    def __init__(self, K: Optional[np.ndarray] = None,
                 kp: float = 0.5,
                 name: str = "P-Controller"):
        """
        Initialize proportional controller.

        Args:
            K: Gain matrix (n_inputs x n_states). If None, uses kp * I.
            kp: Scalar proportional gain (used if K is None)
            name: Controller name
        """
        super().__init__(name)
        self._K = K
        self._kp = kp

        # Lyapunov matrix P for V(e) = e' @ P @ e
        # Using P = I for simplicity (positive definite)
        self._P = None

    @property
    def K(self) -> np.ndarray:
        """Get gain matrix."""
        if self._K is not None:
            return self._K
        # Default: scalar gain times identity
        if self.model is not None:
            return self._kp * np.eye(self.model.n_inputs, self.model.n_states)
        return self._kp * np.eye(2)  # Fallback for 2D integrator

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute proportional control action.

        u = -K @ (x - target)

        Args:
            x: Current state
            t: Current time (not used in P control)

        Returns:
            u: Control input
            diagnostics: Contains error, Lyapunov value, etc.
        """
        x = np.asarray(x, dtype=float)

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        # Compute error
        error = x - self._target

        # Proportional control
        u = -self.K @ error

        # Compute Lyapunov function V(e) = e' @ P @ e
        P = self._get_lyapunov_matrix()
        V = error @ P @ error

        # Estimate Lyapunov derivative (discrete: ΔV = V(e+) - V(e))
        # For analysis, we compute V at current error
        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error),
            'lyapunov': V,
            'gain_matrix': self.K.copy()
        }

        return u, diagnostics

    def _get_lyapunov_matrix(self) -> np.ndarray:
        """Get Lyapunov matrix P for V(e) = e' @ P @ e."""
        if self._P is not None:
            return self._P

        # Default: identity matrix (always positive definite)
        n = self._target.shape[0] if self._target is not None else 2
        return np.eye(n)

    def set_lyapunov_matrix(self, P: np.ndarray) -> None:
        """
        Set custom Lyapunov matrix.

        Args:
            P: Positive definite matrix for V(e) = e' @ P @ e
        """
        # Verify P is symmetric positive definite
        if not np.allclose(P, P.T):
            raise ValueError("P must be symmetric")
        eigvals = np.linalg.eigvalsh(P)
        if np.any(eigvals <= 0):
            raise ValueError("P must be positive definite")
        self._P = P

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value.

        V(x) = (x - target)' @ P @ (x - target)

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._target is None:
            return None
        error = np.asarray(x) - self._target
        P = self._get_lyapunov_matrix()
        return float(error @ P @ error)

    def analyze_stability(self, tau: float) -> Dict[str, Any]:
        """
        Analyze closed-loop stability.

        For integrator: x(t+1) = x(t) + τ*u = x(t) - τ*K*e = (I - τK)*e + target

        Stability requires eigenvalues of (I - τK) inside unit circle.

        Args:
            tau: Sampling period

        Returns:
            Dictionary with stability analysis results
        """
        K = self.K
        n = K.shape[1]

        # Closed-loop matrix for error dynamics
        A_cl = np.eye(n) - tau * K

        # Eigenvalue analysis
        eigvals = np.linalg.eigvals(A_cl)
        spectral_radius = np.max(np.abs(eigvals))

        is_stable = spectral_radius < 1.0

        return {
            'closed_loop_matrix': A_cl,
            'eigenvalues': eigvals,
            'spectral_radius': spectral_radius,
            'is_stable': is_stable,
            'stability_margin': 1.0 - spectral_radius if is_stable else spectral_radius - 1.0
        }
