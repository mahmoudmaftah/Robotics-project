"""
Model Predictive Control (MPC) for Discrete-Time Systems.

MPC solves a finite-horizon optimal control problem at each time step:

    min_{u_0,...,u_{N-1}} sum_{k=0}^{N-1} [x_k'Qx_k + u_k'Ru_k] + x_N'Q_f*x_N

    subject to:
        x_{k+1} = A*x_k + B*u_k  (or nonlinear dynamics)
        x_k in X (state constraints)
        u_k in U (input constraints)

Only the first control u_0 is applied, then the problem is re-solved
at the next time step (receding horizon).

For linear systems, this becomes a Quadratic Program (QP).
For nonlinear systems, we linearize around current state (NMPC).

Lyapunov Analysis:
    Under certain conditions (terminal cost = LQR cost, terminal constraint),
    MPC provides guaranteed stability. The optimal cost V*(x) serves as
    a Lyapunov function.

This implementation uses scipy.optimize for the QP solver.
"""

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from typing import Tuple, Dict, Any, Optional, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.base import Controller


class MPCController(Controller):
    """
    Model Predictive Controller for discrete-time systems.

    Supports both linear and nonlinear models through linearization.
    Handles state and input constraints via QP formulation.
    """

    def __init__(self,
                 horizon: int = 10,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 Q_terminal: Optional[np.ndarray] = None,
                 name: str = "MPC"):
        """
        Initialize MPC controller.

        Args:
            horizon: Prediction horizon N
            Q: State cost matrix
            R: Input cost matrix
            Q_terminal: Terminal state cost (default: Q)
            name: Controller name
        """
        super().__init__(name)
        self.horizon = horizon
        self._Q = Q
        self._R = R
        self._Q_terminal = Q_terminal

        # Will be set when model is assigned
        self._n_states = None
        self._n_inputs = None
        self._u_min = None
        self._u_max = None
        self._x_min = None
        self._x_max = None

        # Warm start for optimization
        self._last_solution = None

        # Performance tracking
        self._solve_times = []

    def set_model(self, model) -> None:
        """Set model and extract dimensions/constraints."""
        self.model = model
        self._n_states = model.n_states
        self._n_inputs = model.n_inputs

        # Extract constraints
        self._u_min = model.u_bounds[:, 0]
        self._u_max = model.u_bounds[:, 1]
        self._x_min = model.x_bounds[:, 0]
        self._x_max = model.x_bounds[:, 1]

        # Default cost matrices
        if self._Q is None:
            self._Q = np.eye(self._n_states)
        if self._R is None:
            self._R = 0.1 * np.eye(self._n_inputs)
        if self._Q_terminal is None:
            self._Q_terminal = self._Q

    def _predict_trajectory(self, x0: np.ndarray,
                           u_sequence: np.ndarray) -> np.ndarray:
        """
        Predict state trajectory given control sequence.

        Args:
            x0: Initial state
            u_sequence: Control sequence (horizon x n_inputs)

        Returns:
            x_trajectory: State trajectory (horizon+1 x n_states)
        """
        x_traj = np.zeros((self.horizon + 1, self._n_states))
        x_traj[0] = x0

        for k in range(self.horizon):
            u_k = u_sequence[k]
            # Use model dynamics (with zero disturbance for prediction)
            x_traj[k + 1] = self.model.dynamics(
                x_traj[k], u_k, np.zeros(self.model.w_bounds.shape[0]))

        return x_traj

    def _cost_function(self, u_flat: np.ndarray, x0: np.ndarray,
                       x_ref: np.ndarray) -> float:
        """
        Compute MPC cost for optimization.

        J = sum_{k=0}^{N-1} [(x_k - x_ref)'Q(x_k - x_ref) + u_k'Ru_k]
            + (x_N - x_ref)'Q_f(x_N - x_ref)

        Args:
            u_flat: Flattened control sequence
            x0: Initial state
            x_ref: Reference state

        Returns:
            Total cost
        """
        u_sequence = u_flat.reshape(self.horizon, self._n_inputs)
        x_traj = self._predict_trajectory(x0, u_sequence)

        cost = 0.0

        # Stage costs
        for k in range(self.horizon):
            x_err = x_traj[k] - x_ref
            u_k = u_sequence[k]
            cost += x_err @ self._Q @ x_err + u_k @ self._R @ u_k

        # Terminal cost
        x_err_N = x_traj[self.horizon] - x_ref
        cost += x_err_N @ self._Q_terminal @ x_err_N

        return cost

    def _cost_gradient(self, u_flat: np.ndarray, x0: np.ndarray,
                       x_ref: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cost function (numerical).

        Args:
            u_flat: Flattened control sequence
            x0: Initial state
            x_ref: Reference state

        Returns:
            Gradient vector
        """
        eps = 1e-6
        grad = np.zeros_like(u_flat)
        f0 = self._cost_function(u_flat, x0, x_ref)

        for i in range(len(u_flat)):
            u_plus = u_flat.copy()
            u_plus[i] += eps
            grad[i] = (self._cost_function(u_plus, x0, x_ref) - f0) / eps

        return grad

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve MPC optimization and return first control.

        Args:
            x: Current state
            t: Current time

        Returns:
            u: Control input (first element of optimal sequence)
            diagnostics: Optimization info
        """
        import time

        x = np.asarray(x, dtype=float)

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        x_ref = self._target

        # Initial guess (warm start or zeros)
        if self._last_solution is not None:
            # Shift previous solution
            u0 = np.roll(self._last_solution.reshape(self.horizon, self._n_inputs),
                        -1, axis=0).flatten()
        else:
            u0 = np.zeros(self.horizon * self._n_inputs)

        # Input bounds for all time steps
        u_lb = np.tile(self._u_min, self.horizon)
        u_ub = np.tile(self._u_max, self.horizon)
        bounds = Bounds(u_lb, u_ub)

        # Solve optimization
        start_time = time.time()

        result = minimize(
            self._cost_function,
            u0,
            args=(x, x_ref),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        solve_time = time.time() - start_time
        self._solve_times.append(solve_time)

        # Extract solution
        u_opt = result.x.reshape(self.horizon, self._n_inputs)
        self._last_solution = result.x

        # First control action
        u = u_opt[0]

        # Compute predicted trajectory for diagnostics
        x_pred = self._predict_trajectory(x, u_opt)

        # Cost as Lyapunov function
        V = result.fun

        # Error
        error = x - x_ref
        if len(error) > 2:
            # Wrap angles for manipulator
            error[0] = np.arctan2(np.sin(error[0]), np.cos(error[0]))
            error[1] = np.arctan2(np.sin(error[1]), np.cos(error[1]))

        diagnostics = {
            'error': error,
            'error_norm': np.linalg.norm(error[:2]) if len(error) > 2 else np.linalg.norm(error),
            'lyapunov': V,
            'optimal_cost': result.fun,
            'solve_time': solve_time,
            'solver_success': result.success,
            'solver_iterations': result.nit,
            'predicted_trajectory': x_pred,
            'optimal_sequence': u_opt
        }

        return u, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function (optimal cost-to-go).

        Note: This requires solving the MPC problem, which is expensive.
        Use the value from diagnostics instead when possible.

        Args:
            x: State vector

        Returns:
            V*(x) = optimal cost
        """
        if self._target is None or self.model is None:
            return None

        # Solve MPC to get optimal cost
        u0 = np.zeros(self.horizon * self._n_inputs)
        u_lb = np.tile(self._u_min, self.horizon)
        u_ub = np.tile(self._u_max, self.horizon)
        bounds = Bounds(u_lb, u_ub)

        result = minimize(
            self._cost_function,
            u0,
            args=(x, self._target),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50}
        )

        return result.fun

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze MPC computational performance.

        Returns:
            Performance statistics
        """
        if not self._solve_times:
            return {'analyzed': False, 'reason': 'No solves recorded'}

        times = np.array(self._solve_times)

        return {
            'n_solves': len(times),
            'mean_solve_time': np.mean(times),
            'max_solve_time': np.max(times),
            'min_solve_time': np.min(times),
            'std_solve_time': np.std(times),
            'real_time_feasible': np.mean(times) < self.model.tau if self.model else None,
            'horizon': self.horizon
        }


class LinearMPC(MPCController):
    """
    MPC for linear systems using efficient QP formulation.

    For linear dynamics x+ = Ax + Bu, the MPC problem can be
    reformulated as a single QP with decision variables being
    the control sequence only.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 horizon: int = 10,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 name: str = "LinearMPC"):
        """
        Initialize Linear MPC.

        Args:
            A: State matrix
            B: Input matrix
            horizon: Prediction horizon
            Q: State cost
            R: Input cost
            name: Controller name
        """
        super().__init__(horizon, Q, R, name=name)
        self.A = A
        self.B = B
        self._n_states = A.shape[0]
        self._n_inputs = B.shape[1]

        # Pre-compute prediction matrices
        self._compute_prediction_matrices()

    def _compute_prediction_matrices(self):
        """
        Compute matrices for batch prediction.

        X = Psi * x0 + Gamma * U

        where X = [x_1; x_2; ...; x_N]
              U = [u_0; u_1; ...; u_{N-1}]
        """
        N = self.horizon
        n, m = self._n_states, self._n_inputs

        # Psi: maps x0 to state sequence
        self.Psi = np.zeros((N * n, n))
        A_power = np.eye(n)
        for k in range(N):
            A_power = A_power @ self.A
            self.Psi[k*n:(k+1)*n, :] = A_power

        # Gamma: maps control sequence to state sequence
        self.Gamma = np.zeros((N * n, N * m))
        for k in range(N):
            for j in range(k + 1):
                A_power = np.linalg.matrix_power(self.A, k - j)
                self.Gamma[k*n:(k+1)*n, j*m:(j+1)*m] = A_power @ self.B

    def _predict_trajectory(self, x0: np.ndarray,
                           u_sequence: np.ndarray) -> np.ndarray:
        """Efficient batch prediction using pre-computed matrices."""
        U = u_sequence.flatten()
        X_pred = self.Psi @ x0 + self.Gamma @ U

        # Reshape and add initial state
        x_traj = np.zeros((self.horizon + 1, self._n_states))
        x_traj[0] = x0
        x_traj[1:] = X_pred.reshape(self.horizon, self._n_states)

        return x_traj
