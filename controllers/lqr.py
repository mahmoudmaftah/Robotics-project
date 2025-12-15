"""
Linear Quadratic Regulator (LQR) Controller.

Optimal state-feedback controller minimizing:
    J = Σ (x'Qx + u'Ru)

For discrete-time systems, solves the discrete algebraic Riccati equation (DARE).

Lyapunov analysis:
    The LQR solution P from DARE serves as a Lyapunov function:
    V(x) = x' @ P @ x

    The closed-loop system satisfies:
    V(x(t+1)) - V(x(t)) = -x'(Q + K'RK)x < 0  for x ≠ 0

    This proves asymptotic stability of the closed-loop system.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Dict, Any, Optional
from .base import Controller


class LQRController(Controller):
    """
    Discrete-time Linear Quadratic Regulator.

    Computes optimal gain K that minimizes quadratic cost.
    Uses scipy.linalg.solve_discrete_are for DARE solution.
    """

    def __init__(self,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 name: str = "LQR"):
        """
        Initialize LQR controller.

        Args:
            Q: State cost matrix (positive semi-definite)
            R: Input cost matrix (positive definite)
            name: Controller name
        """
        super().__init__(name)
        self._Q = Q
        self._R = R
        self._K = None  # Computed gain
        self._P = None  # DARE solution (Lyapunov matrix)
        self._A = None  # System matrices
        self._B = None

    def design(self, A: np.ndarray, B: np.ndarray,
               Q: Optional[np.ndarray] = None,
               R: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Design LQR controller for given system.

        Solves discrete algebraic Riccati equation:
            A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0

        Optimal gain:
            K = (R + B'PB)^{-1} B'PA

        Args:
            A: State matrix (n x n)
            B: Input matrix (n x m)
            Q: State cost (default: identity)
            R: Input cost (default: identity)

        Returns:
            K: Optimal feedback gain matrix
        """
        n = A.shape[0]
        m = B.shape[1]

        if Q is None:
            Q = self._Q if self._Q is not None else np.eye(n)
        if R is None:
            R = self._R if self._R is not None else np.eye(m)

        self._Q = Q
        self._R = R
        self._A = A
        self._B = B

        # Solve discrete algebraic Riccati equation
        try:
            P = linalg.solve_discrete_are(A, B, Q, R)
        except np.linalg.LinAlgError:
            # Fallback: use continuous ARE and discretize
            raise ValueError("DARE solution failed. System may not be stabilizable.")

        self._P = P

        # Compute optimal gain: K = (R + B'PB)^{-1} B'PA
        self._K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        return self._K

    def design_for_model(self, model, x_eq: Optional[np.ndarray] = None,
                         u_eq: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Design LQR for a robot model.

        Gets linearized system from model and designs controller.

        Args:
            model: RobotModel instance
            x_eq: Equilibrium state (default: zeros)
            u_eq: Equilibrium input (default: zeros)

        Returns:
            K: Optimal gain matrix
        """
        self.model = model

        if x_eq is None:
            x_eq = np.zeros(model.n_states)
        if u_eq is None:
            u_eq = np.zeros(model.n_inputs)

        A, B = model.get_linearization(x_eq, u_eq)
        return self.design(A, B)

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute LQR control action.

        u = -K @ (x - target)

        Args:
            x: Current state
            t: Current time (not used)

        Returns:
            u: Control input
            diagnostics: Contains error, Lyapunov value, cost info
        """
        x = np.asarray(x, dtype=float)

        if self._K is None:
            raise ValueError("Controller not designed. Call design() first.")

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        # Compute error
        error = x - self._target

        # LQR control
        u = -self._K @ error

        # Compute Lyapunov function V(e) = e' @ P @ e
        V = error @ self._P @ error

        # Stage cost: x'Qx + u'Ru
        stage_cost = error @ self._Q @ error + u @ self._R @ u

        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error),
            'lyapunov': V,
            'stage_cost': stage_cost,
            'gain_matrix': self._K.copy()
        }

        return u, diagnostics

    @property
    def K(self) -> Optional[np.ndarray]:
        """Get LQR gain matrix."""
        return self._K

    @property
    def P(self) -> Optional[np.ndarray]:
        """Get DARE solution (Lyapunov matrix)."""
        return self._P

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value.

        V(x) = (x - target)' @ P @ (x - target)

        where P is the solution to DARE.

        Args:
            x: State vector

        Returns:
            V(x) value
        """
        if self._P is None or self._target is None:
            return None
        error = np.asarray(x) - self._target
        return float(error @ self._P @ error)

    def verify_lyapunov_decrease(self, x: np.ndarray, x_next: np.ndarray) -> Dict[str, Any]:
        """
        Verify Lyapunov function decrease along trajectory.

        For LQR, we should have:
            ΔV = V(x+) - V(x) = -x'(Q + K'RK)x < 0

        Args:
            x: Current state
            x_next: Next state

        Returns:
            Dictionary with verification results
        """
        if self._P is None or self._target is None:
            return {'verified': False, 'reason': 'Controller not designed'}

        e = x - self._target
        e_next = x_next - self._target

        V = e @ self._P @ e
        V_next = e_next @ self._P @ e_next

        delta_V = V_next - V

        # Theoretical decrease
        theoretical_decrease = -e @ (self._Q + self._K.T @ self._R @ self._K) @ e

        return {
            'V_current': V,
            'V_next': V_next,
            'delta_V': delta_V,
            'theoretical_delta_V': theoretical_decrease,
            'is_decreasing': delta_V < 0,
            'verified': delta_V <= 0 or np.isclose(np.linalg.norm(e), 0)
        }

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze closed-loop stability.

        Returns eigenvalues and stability margins.
        """
        if self._A is None or self._K is None:
            return {'analyzed': False, 'reason': 'Controller not designed'}

        # Closed-loop system: x+ = (A - BK)x
        A_cl = self._A - self._B @ self._K

        eigvals = np.linalg.eigvals(A_cl)
        spectral_radius = np.max(np.abs(eigvals))

        return {
            'closed_loop_matrix': A_cl,
            'eigenvalues': eigvals,
            'spectral_radius': spectral_radius,
            'is_stable': spectral_radius < 1.0,
            'gain_margin': 1.0 / spectral_radius if spectral_radius > 0 else np.inf
        }
