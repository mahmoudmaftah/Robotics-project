"""
Computed Torque Controller for Two-Link Manipulator.

Also known as inverse dynamics or feedback linearization for manipulators.

The manipulator dynamics:
    M(q)q_ddot + C(q, q_dot)q_dot + g(q) = tau

Computed torque control:
    tau = M(q)(q_ddot_d + Kd*(q_dot_d - q_dot) + Kp*(q_d - q)) + C(q,q_dot)q_dot + g(q)

This results in closed-loop dynamics:
    e_ddot + Kd*e_dot + Kp*e = 0

which is a linear second-order system with eigenvalues determined by Kp, Kd.

Lyapunov Analysis:
    Define: s = e_dot + lambda*e
    V = 0.5 * s' * M(q) * s + 0.5 * e' * Kp * e

    With proper choice of Kp, Kd:
    V_dot = -s' * Kd * s <= 0

    Asymptotic stability follows from LaSalle's invariance principle.

Note: This implementation uses Euler discretization. The computed torque
is computed based on current model knowledge.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from controllers.base import Controller


class ComputedTorqueController(Controller):
    """
    Computed torque (inverse dynamics) controller for two-link manipulator.

    Provides exact linearization of manipulator dynamics through
    model-based compensation.
    """

    def __init__(self, Kp: Optional[np.ndarray] = None,
                 Kd: Optional[np.ndarray] = None,
                 name: str = "ComputedTorque"):
        """
        Initialize computed torque controller.

        Args:
            Kp: Position gain matrix (2x2, default: 100*I)
            Kd: Velocity gain matrix (2x2, default: 20*I)
            name: Controller name

        Note: For critically damped response with natural frequency wn,
              choose Kp = wn^2 * I, Kd = 2*wn * I
        """
        super().__init__(name)

        # Default gains for wn = 10 rad/s, critically damped
        self._Kp = Kp if Kp is not None else 100.0 * np.eye(2)
        self._Kd = Kd if Kd is not None else 20.0 * np.eye(2)

        # Desired trajectory (for tracking)
        self._q_d = None      # Desired position
        self._q_dot_d = None  # Desired velocity
        self._q_ddot_d = None # Desired acceleration

    def set_target(self, target: np.ndarray) -> None:
        """
        Set target joint configuration.

        Args:
            target: Target state [theta1, theta2, theta1_dot, theta2_dot]
                   or just [theta1, theta2] with zero velocity
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
            raise ValueError("Target must be 2D (position) or 4D (position + velocity)")

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute computed torque control.

        tau = M(q)(q_ddot_d + Kd*e_dot + Kp*e) + C(q,q_dot)q_dot + g(q)

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

        # Extract current state
        q = x[:2]      # Joint positions
        q_dot = x[2:]  # Joint velocities

        # Compute errors
        e = self._q_d - q          # Position error
        e_dot = self._q_dot_d - q_dot  # Velocity error

        # Wrap angle errors to [-pi, pi]
        e[0] = np.arctan2(np.sin(e[0]), np.cos(e[0]))
        e[1] = np.arctan2(np.sin(e[1]), np.cos(e[1]))

        # Get model matrices
        M = self.model.mass_matrix(q)
        c = self.model.coriolis_vector(q, q_dot)
        g = self.model.gravity_vector(q)

        # Computed torque control law
        # Desired acceleration in closed-loop
        q_ddot_ref = self._q_ddot_d + self._Kd @ e_dot + self._Kp @ e

        # Control torque
        tau = M @ q_ddot_ref + c + g

        # Lyapunov function (energy-based)
        # V = 0.5 * e_dot' * M * e_dot + 0.5 * e' * Kp * e
        V = 0.5 * (e_dot @ M @ e_dot + e @ self._Kp @ e)

        diagnostics = {
            'error': np.concatenate([e, e_dot]),
            'error_norm': np.linalg.norm(e),
            'velocity_error_norm': np.linalg.norm(e_dot),
            'lyapunov': V,
            'position_error': e.copy(),
            'velocity_error': e_dot.copy(),
            'mass_matrix': M.copy(),
            'coriolis': c.copy(),
            'gravity': g.copy()
        }

        return tau, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute energy-based Lyapunov function.

        V = 0.5 * e_dot' * M(q) * e_dot + 0.5 * e' * Kp * e

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
        e_dot = self._q_dot_d - q_dot

        # Wrap angles
        e[0] = np.arctan2(np.sin(e[0]), np.cos(e[0]))
        e[1] = np.arctan2(np.sin(e[1]), np.cos(e[1]))

        M = self.model.mass_matrix(q)
        return 0.5 * float(e_dot @ M @ e_dot + e @ self._Kp @ e)

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze closed-loop stability.

        The linearized error dynamics are:
            e_ddot + Kd*e_dot + Kp*e = 0

        This is stable if Kp > 0 and Kd > 0.

        Returns:
            Stability analysis results
        """
        # Eigenvalues of error dynamics
        # Characteristic equation: s^2 + Kd*s + Kp = 0
        # For diagonal Kp, Kd, eigenvalues are independent per joint

        results = {}

        for i in range(2):
            kp_i = self._Kp[i, i]
            kd_i = self._Kd[i, i]

            # Eigenvalues: s = (-kd +/- sqrt(kd^2 - 4*kp)) / 2
            discriminant = kd_i**2 - 4*kp_i

            if discriminant >= 0:
                s1 = (-kd_i + np.sqrt(discriminant)) / 2
                s2 = (-kd_i - np.sqrt(discriminant)) / 2
                damping = 'overdamped' if discriminant > 0 else 'critically_damped'
            else:
                real_part = -kd_i / 2
                imag_part = np.sqrt(-discriminant) / 2
                s1 = complex(real_part, imag_part)
                s2 = complex(real_part, -imag_part)
                damping = 'underdamped'

            wn = np.sqrt(kp_i)  # Natural frequency
            zeta = kd_i / (2 * wn)  # Damping ratio

            results[f'joint_{i+1}'] = {
                'eigenvalues': [s1, s2],
                'natural_frequency': wn,
                'damping_ratio': zeta,
                'damping_type': damping,
                'is_stable': kp_i > 0 and kd_i > 0
            }

        results['globally_stable'] = all(r['is_stable'] for r in results.values()
                                         if isinstance(r, dict))
        results['note'] = 'Exact linearization - stability holds globally with accurate model'

        return results
