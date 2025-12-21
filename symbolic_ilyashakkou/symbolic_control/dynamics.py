"""
Dynamics Module - Abstract base class and concrete implementations.

This module isolates all model-specific math. To add a new robot model,
simply create a new class inheriting from Dynamics.
"""

from abc import ABC, abstractmethod
import numpy as np


class Dynamics(ABC):
    @property
    def disturbance_dim(self) -> int:
        """Dimension of the disturbance vector. Override if different from state_dim."""
        return self.state_dim

    @property
    def w_min(self) -> np.ndarray:
        """Minimum disturbance (vector)."""
        return -self.w_bound * np.ones(self.disturbance_dim)

    @property
    def w_max(self) -> np.ndarray:
        """Maximum disturbance (vector)."""
        return self.w_bound * np.ones(self.disturbance_dim)
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
        self.w_bound = w_bound
        
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


class UnicycleDynamics(Dynamics):
    @property
    def disturbance_dim(self) -> int:
        return 3
    """
    3D Unicycle model (mobile robot with heading).
    
    Dynamics:
        x1(t+1) = x1(t) + τ * (u1 * cos(x3(t)) + w1)
        x2(t+1) = x2(t) + τ * (u1 * sin(x3(t)) + w2)
        x3(t+1) = x3(t) + τ * (u2 + w3)  (mod 2π, wrapped to [-π, π])
    
    where:
        - (x1, x2) is position, x3 is heading angle
        - u1 is linear velocity, u2 is angular velocity
        - w = (w1, w2, w3) is disturbance
    
    This system is NON-MONOTONE due to sin/cos terms.
    We use Jacobian-based growth bounds for over-approximation.
    """
    
    def __init__(self, tau=1.0, w_bound=0.05, v_values=None, omega_values=None):
        """
        Args:
            tau: Sampling period
            w_bound: Disturbance bound (symmetric: w ∈ [-w_bound, w_bound]^3)
            v_values: List of linear velocity values (u1 ∈ [0.25, 1])
            omega_values: List of angular velocity values (u2 ∈ [-1, 1])
        """
        self.tau = tau
        self.w_bound = w_bound
        
        # Default control discretization
        if v_values is None:
            v_values = [0.25, 0.5, 0.75, 1.0]
        if omega_values is None:
            omega_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        self.v_values = np.array(v_values)
        self.omega_values = np.array(omega_values)
        self.v_max = np.max(np.abs(v_values))
        
        # Build control set: all combinations of (v, omega)
        self._control_set = [
            np.array([v, omega]) 
            for v in v_values 
            for omega in omega_values
        ]
    
    @property
    def state_dim(self) -> int:
        return 3
    
    @property
    def control_set(self) -> list:
        return self._control_set
    
    def _wrap_angle(self, theta: float) -> float:
        """Wrap angle to [-π, π]."""
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Exact state update for simulation.
        
        Args:
            x: State [x1, x2, x3] (position and heading)
            u: Control [u1, u2] (linear and angular velocity)
            w: Disturbance [w1, w2, w3]
            
        Returns:
            Next state with angle wrapped to [-π, π]
        """
        x1, x2, x3 = x
        u1, u2 = u
        w1, w2, w3 = w
        
        x1_next = x1 + self.tau * (u1 * np.cos(x3) + w1)
        x2_next = x2 + self.tau * (u1 * np.sin(x3) + w2)
        x3_next = self._wrap_angle(x3 + self.tau * (u2 + w3))
        
        return np.array([x1_next, x2_next, x3_next])
    
    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Over-approximation using Jacobian-based growth bounds.
        
        For non-monotone systems, we use:
            successor ∈ [f(x*, u, w*) - Δ, f(x*, u, w*) + Δ]
        where:
            x* = center of cell
            w* = center of disturbance (= 0 for symmetric bounds)
            Δ = D_x * δ_x + D_w * δ_w
            D_x, D_w = bounds on Jacobians
            δ_x, δ_w = half-widths of state cell and disturbance
        
        Jacobian analysis for unicycle:
            ∂f/∂x = [[1, 0, -τ*u1*sin(x3)],
                     [0, 1,  τ*u1*cos(x3)],
                     [0, 0,  1]]
            
            |∂f/∂x| ≤ [[1, 0, τ*|u1|],
                       [0, 1, τ*|u1|],
                       [0, 0, 1]]
            
            ∂f/∂w = τ * I_3
        """
        u1, _u2 = u
        
        # Cell center and half-width
        x_center = (x_lo + x_hi) / 2
        delta_x = (x_hi - x_lo) / 2
        
        # Disturbance center (0 for symmetric) and half-width
        delta_w = np.array([self.w_bound, self.w_bound, self.w_bound])
        w_center = np.zeros(3)
        
        # Compute center transition
        f_center = self.step(x_center, u, w_center)
        
        # Jacobian bounds (element-wise absolute values)
        # d_x: bound on |∂f_i/∂x_j|
        # For x3 derivative: |∂f1/∂x3| = τ*|u1|*|sin(x3)| ≤ τ*|u1|
        #                    |∂f2/∂x3| = τ*|u1|*|cos(x3)| ≤ τ*|u1|
        d_x = np.array([
            [1.0, 0.0, self.tau * np.abs(u1)],
            [0.0, 1.0, self.tau * np.abs(u1)],
            [0.0, 0.0, 1.0]
        ])
        
        # d_w: bound on |∂f_i/∂w_j| = τ * I_3
        d_w = self.tau * np.eye(3)
        
        # Growth bound: Δ = d_x @ δ_x + d_w @ δ_w
        growth = d_x @ delta_x + d_w @ delta_w
        
        # Over-approximation bounds
        succ_lo = f_center - growth
        succ_hi = f_center + growth
        
        # Wrap angle bounds (conservative: if interval crosses ±π, expand to full range)
        theta_lo, theta_hi = succ_lo[2], succ_hi[2]
        if theta_hi - theta_lo >= 2 * np.pi:
            # Interval covers full circle
            succ_lo[2] = -np.pi
            succ_hi[2] = np.pi
        else:
            # Wrap both bounds
            theta_lo_wrapped = self._wrap_angle(theta_lo)
            theta_hi_wrapped = self._wrap_angle(theta_hi)
            
            # Check if wrapping caused inversion (crossed ±π boundary)
            if theta_lo_wrapped > theta_hi_wrapped:
                # Conservative: expand to full range
                succ_lo[2] = -np.pi
                succ_hi[2] = np.pi
            else:
                succ_lo[2] = theta_lo_wrapped
                succ_hi[2] = theta_hi_wrapped
        
        return succ_lo, succ_hi

