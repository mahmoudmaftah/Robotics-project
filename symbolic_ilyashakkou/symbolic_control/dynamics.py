"""
Dynamics Module - Abstract base class and concrete implementations.

This module isolates all model-specific math. To add a new robot model,
simply create a new class inheriting from Dynamics.
"""

from abc import ABC, abstractmethod
import numpy as np


class Dynamics(ABC):
    """
    Abstract base class for dynamical systems.
    
    Any concrete model must implement:
    - post(): Over-approximated successor set (for abstraction)
    - step(): Exact next state (for simulation)
    - state_dim, control_set: System properties
    """
    
    @abstractmethod
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Compute over-approximated reachable set from cell [x_lo, x_hi] under input u.
        
        Args:
            x_lo: Lower-left corner of the cell
            x_hi: Upper-right corner of the cell
            u: Control input
            
        Returns:
            (succ_lo, succ_hi): Bounding box of all possible successors
        """
        pass
    
    @abstractmethod
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute exact next state (for simulation).
        
        Args:
            x: Current state
            u: Control input
            w: Disturbance
            
        Returns:
            Next state
        """
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state space."""
        pass
    
    @property
    @abstractmethod
    def control_set(self) -> list:
        """List of all discrete control inputs."""
        pass


class IntegratorDynamics(Dynamics):
    """
    2D Integrator (velocity-controlled point robot).
    
    Dynamics:
        x1(t+1) = x1(t) + τ * (u1 + w1)
        x2(t+1) = x2(t) + τ * (u2 + w2)
    
    This system is MONOTONE: increasing x or u always increases the successor.
    This allows exact interval arithmetic (no sampling needed).
    """
    
    def __init__(self, tau=1.0, w_bound=0.05, u_values=None):
        """
        Args:
            tau: Sampling period
            w_bound: Disturbance bound (symmetric: w ∈ [-w_bound, w_bound])
            u_values: List of discrete control values per axis (default: [-1, -0.5, 0, 0.5, 1])
        """
        self.tau = tau
        self.w_min = -w_bound
        self.w_max = w_bound
        
        # Default control discretization
        if u_values is None:
            u_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Build control set: all combinations of (u1, u2)
        self._control_set = [
            np.array([u1, u2]) 
            for u1 in u_values 
            for u2 in u_values
        ]
    
    @property
    def state_dim(self) -> int:
        return 2
    
    @property
    def control_set(self) -> list:
        return self._control_set
    
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Over-approximation using MONOTONICITY.
        
        Since f(x, u, w) = x + τ(u + w) is monotone increasing in x and w:
        - Minimum successor: f(x_lo, u, w_min)
        - Maximum successor: f(x_hi, u, w_max)
        """
        w_lo = np.array([self.w_min, self.w_min])
        w_hi = np.array([self.w_max, self.w_max])
        
        succ_lo = self.step(x_lo, u, w_lo)
        succ_hi = self.step(x_hi, u, w_hi)
        
        return succ_lo, succ_hi
    
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Exact state update for simulation."""
        return x + self.tau * (u + w)
