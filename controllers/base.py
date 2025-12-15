"""
Base controller interface.

All controllers must implement compute_control(x, t) -> (u, diagnostics).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


class Controller(ABC):
    """
    Abstract base class for all controllers.

    Controllers compute control actions given the current state and time.
    They must implement the compute_control method which returns both
    the control input and optional diagnostic information.

    Attributes:
        name: Human-readable controller name
        model: Reference to the robot model (for constraints)
    """

    def __init__(self, name: str = "BaseController"):
        """
        Initialize controller.

        Args:
            name: Controller identifier for logging/plotting
        """
        self.name = name
        self.model = None
        self._target = None

    def set_model(self, model) -> None:
        """
        Associate controller with a robot model.

        Args:
            model: RobotModel instance for constraint information
        """
        self.model = model

    def set_target(self, target: np.ndarray) -> None:
        """
        Set the control target (goal state).

        Args:
            target: Target state vector
        """
        self._target = np.asarray(target, dtype=float)

    @property
    def target(self) -> Optional[np.ndarray]:
        """Get current target state."""
        return self._target

    @abstractmethod
    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute control action.

        Args:
            x: Current state vector
            t: Current time (seconds)

        Returns:
            u: Control input vector (raw, before saturation)
            diagnostics: Dictionary with controller-specific info, e.g.:
                - 'error': tracking error
                - 'lyapunov': Lyapunov function value V(x)
                - 'lyapunov_derivative': ΔV or dV/dt estimate
        """
        pass

    def reset(self) -> None:
        """Reset controller state (for controllers with memory, e.g., PID)."""
        pass

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function value V(x).

        Override in subclasses that provide stability certificates.

        Args:
            x: State vector

        Returns:
            V(x) if available, None otherwise
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
