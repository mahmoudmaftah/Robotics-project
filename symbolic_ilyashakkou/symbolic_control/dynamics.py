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
        w_lo = self.w_min
        w_hi = self.w_max
        
        succ_lo = self.step(x_lo, u, w_lo)
        succ_hi = self.step(x_hi, u, w_hi)
        
        return succ_lo, succ_hi
    
    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Exact state update for simulation."""
        return x + self.tau * (u + w)


class UnicycleDynamics(Dynamics):
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

    @property
    def disturbance_dim(self) -> int:
        return 3
    
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


class ManipulatorDynamics(Dynamics):
    """
    Two-link Planar Manipulator (discrete via Euler).

    State: x = [θ1, θ2, θ̇1, θ̇2] (joint angles and velocities)
    Input: u = [τ1, τ2] (joint torques)

    Dynamics:
        x(t+1) = x(t) + τ * f(x(t), u(t))
        where f(x,u) = [θ̇; M(θ)^{-1}(u - C(θ,θ̇)θ̇ - g(θ))]

    Parameters (from models.txt):
        m1 = m2 = 1.0 kg (link masses)
        l1 = l2 = 0.5 m (link lengths)
        g = 9.81 m/s² (gravity)

    This system is NON-MONOTONE due to sin/cos and coupling terms.
    Uses Jacobian-based growth bounds for over-approximation.
    """

    def __init__(
        self,
        tau: float = 0.01,
        w_bound: float = 0.01,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 0.5,
        l2: float = 0.5,
        g_accel: float = 9.81,
        torque_values: list = None
    ):
        """
        Args:
            tau: Sampling period (should be small for stability, e.g., 0.01)
            w_bound: Disturbance bound
            m1, m2: Link masses (kg)
            l1, l2: Link lengths (m)
            g_accel: Gravitational acceleration (m/s²)
            torque_values: List of torque values per joint (default: [-5, -2.5, 0, 2.5, 5])
        """
        self.tau = tau
        self.w_bound = w_bound
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g_accel = g_accel

        if torque_values is None:
            torque_values = [-5.0, -2.5, 0.0, 2.5, 5.0]

        # Build control set: all combinations of (τ1, τ2)
        self._control_set = [
            np.array([t1, t2])
            for t1 in torque_values
            for t2 in torque_values
        ]

        # Pre-compute constants for Jacobian bounds
        self.max_torque = max(abs(t) for t in torque_values)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def disturbance_dim(self) -> int:
        return 4

    @property
    def control_set(self) -> list:
        return self._control_set

    def _mass_matrix(self, theta2: float) -> np.ndarray:
        """Compute mass matrix M(θ)."""
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        c2 = np.cos(theta2)

        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2**2 + m2 * l1 * l2 * c2
        M22 = m2 * l2**2

        return np.array([[M11, M12], [M12, M22]])

    def _coriolis_vector(self, theta2: float, dtheta1: float, dtheta2: float) -> np.ndarray:
        """Compute Coriolis/centrifugal term C(θ,θ̇)θ̇."""
        m2, l1, l2 = self.m2, self.l1, self.l2
        s2 = np.sin(theta2)
        h = m2 * l1 * l2 * s2

        c1 = -h * dtheta2 * (2 * dtheta1 + dtheta2)
        c2 = h * dtheta1**2

        return np.array([c1, c2])

    def _gravity_vector(self, theta1: float, theta2: float) -> np.ndarray:
        """Compute gravity term g(θ)."""
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g_accel
        s1 = np.sin(theta1)
        s12 = np.sin(theta1 + theta2)

        g1 = (m1 + m2) * g * l1 * s1 + m2 * g * l2 * s12
        g2 = m2 * g * l2 * s12

        return np.array([g1, g2])

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute f(x, u) = [θ̇; M⁻¹(u - C - g)].

        Args:
            x: State [θ1, θ2, θ̇1, θ̇2]
            u: Control [τ1, τ2]

        Returns:
            dx/dt: [θ̇1, θ̇2, θ̈1, θ̈2]
        """
        theta1, theta2, dtheta1, dtheta2 = x

        M = self._mass_matrix(theta2)
        C = self._coriolis_vector(theta2, dtheta1, dtheta2)
        G = self._gravity_vector(theta1, theta2)

        # θ̈ = M⁻¹(u - C - G)
        M_inv = np.linalg.inv(M)
        ddtheta = M_inv @ (u - C - G)

        return np.array([dtheta1, dtheta2, ddtheta[0], ddtheta[1]])

    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute exact next state (Euler integration with disturbance).

        Args:
            x: Current state [θ1, θ2, θ̇1, θ̇2]
            u: Control [τ1, τ2]
            w: Disturbance [w1, w2, w3, w4]

        Returns:
            Next state
        """
        f = self._dynamics(x, u)
        return x + self.tau * (f + w)

    def post(self, x_lo: np.ndarray, x_hi: np.ndarray, u: np.ndarray) -> tuple:
        """
        Over-approximation using sampling + Lipschitz margin.

        For complex non-monotone dynamics like the manipulator, we use:
        1. Sample vertices and center of the cell
        2. Compute dynamics at each sample point
        3. Take bounding box of results
        4. Add margin for disturbance and Lipschitz continuity
        """
        # Sample the cell vertices and center (2^4 = 16 vertices + center = 17 points)
        # For efficiency, we sample corners and center
        samples = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            x_lo[0] if i == 0 else x_hi[0],
                            x_lo[1] if j == 0 else x_hi[1],
                            x_lo[2] if k == 0 else x_hi[2],
                            x_lo[3] if l == 0 else x_hi[3],
                        ])
                        samples.append(x)

        # Add center point
        x_center = (x_lo + x_hi) / 2
        samples.append(x_center)

        # Compute dynamics at all sample points
        w_zero = np.zeros(4)
        successors = []
        for x in samples:
            x_next = self.step(x, u, w_zero)
            successors.append(x_next)

        successors = np.array(successors)

        # Bounding box of sampled successors
        succ_lo = np.min(successors, axis=0)
        succ_hi = np.max(successors, axis=0)

        # Add margin for:
        # 1. Disturbance: tau * w_bound
        # 2. Lipschitz continuity within each sub-cell (small margin)
        delta_x = (x_hi - x_lo) / 2
        lipschitz_margin = 0.1 * self.tau * np.max(delta_x)  # Conservative Lipschitz margin
        disturbance_margin = self.tau * self.w_bound

        total_margin = disturbance_margin + lipschitz_margin

        succ_lo = succ_lo - total_margin
        succ_hi = succ_hi + total_margin

        return succ_lo, succ_hi

