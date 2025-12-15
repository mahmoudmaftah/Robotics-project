"""
Reinforcement Learning Controller using Policy Gradient.

Implements REINFORCE algorithm (Monte Carlo Policy Gradient) for
learning a control policy from interaction with the environment.

Policy: pi(a|s; theta) = N(mu(s; theta), sigma^2)
    - Gaussian policy with learned mean
    - mu(s) = theta' @ phi(s) (linear in features)
    - phi(s) = [s, 1] (simple feature vector)

Update rule (REINFORCE):
    theta <- theta + alpha * sum_t [grad_theta log pi(a_t|s_t) * G_t]

where G_t = sum_{k=t}^T gamma^{k-t} * r_k is the return from time t.

Reward design:
    r(s, a, s') = -||s' - goal||^2 - lambda * ||a||^2

This implementation is for educational purposes. For production,
use stable-baselines3 or similar libraries.

Note on stability:
    RL does not provide formal stability guarantees. The learned policy
    may not stabilize the system, especially during early training.
    Always verify the learned policy before deployment.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.base import Controller


@dataclass
class Trajectory:
    """Container for a single episode trajectory."""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def total_return(self) -> float:
        return sum(self.rewards)


class PolicyGradientController(Controller):
    """
    Policy Gradient (REINFORCE) controller.

    Learns a Gaussian policy for continuous control.
    """

    def __init__(self,
                 n_states: int = 2,
                 n_actions: int = 2,
                 learning_rate: float = 0.01,
                 gamma: float = 0.99,
                 sigma: float = 0.5,
                 action_cost: float = 0.01,
                 name: str = "PolicyGradient"):
        """
        Initialize policy gradient controller.

        Args:
            n_states: State dimension
            n_actions: Action dimension
            learning_rate: Learning rate alpha
            gamma: Discount factor
            sigma: Policy standard deviation (exploration)
            action_cost: Penalty coefficient for control effort
            name: Controller name
        """
        super().__init__(name)
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.sigma = sigma
        self.action_cost = action_cost

        # Policy parameters: theta is (n_states + 1) x n_actions
        # Linear policy: mu(s) = theta' @ [s; 1]
        self.theta = np.zeros((n_states + 1, n_actions))

        # Training data
        self._trajectories: List[Trajectory] = []
        self._episode_returns: List[float] = []

        # Training state
        self._training_mode = False
        self._current_trajectory: Optional[Trajectory] = None

    def _features(self, state: np.ndarray) -> np.ndarray:
        """
        Compute feature vector from state.

        phi(s) = [s - target, 1] (error-based features)

        Args:
            state: Current state

        Returns:
            Feature vector
        """
        if self._target is not None:
            error = state - self._target
        else:
            error = state
        return np.concatenate([error, [1.0]])

    def _policy_mean(self, state: np.ndarray) -> np.ndarray:
        """
        Compute policy mean mu(s) = theta' @ phi(s).

        Args:
            state: Current state

        Returns:
            Mean action
        """
        phi = self._features(state)
        return self.theta.T @ phi

    def _sample_action(self, state: np.ndarray) -> np.ndarray:
        """
        Sample action from Gaussian policy.

        a ~ N(mu(s), sigma^2 * I)

        Args:
            state: Current state

        Returns:
            Sampled action
        """
        mean = self._policy_mean(state)
        noise = np.random.randn(self.n_actions) * self.sigma
        return mean + noise

    def _log_policy_gradient(self, state: np.ndarray,
                             action: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log policy.

        For Gaussian: grad log pi = (a - mu) / sigma^2 * grad mu
                                  = (a - mu) / sigma^2 * phi(s)

        Args:
            state: State
            action: Action taken

        Returns:
            Gradient w.r.t. theta (flattened)
        """
        phi = self._features(state)
        mean = self._policy_mean(state)

        # grad_theta log pi = outer(phi, (a - mu)) / sigma^2
        grad = np.outer(phi, (action - mean)) / (self.sigma ** 2)
        return grad

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray) -> float:
        """
        Compute reward for transition.

        r = -||next_state - target||^2 - lambda * ||action||^2

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        if self._target is None:
            return 0.0

        # Distance to goal (negative = penalty)
        dist = np.linalg.norm(next_state - self._target)

        # Action cost
        action_penalty = self.action_cost * np.linalg.norm(action) ** 2

        # Reward: closer to goal is better, less control effort is better
        reward = -dist ** 2 - action_penalty

        # Bonus for reaching goal
        if dist < 0.5:
            reward += 10.0

        return reward

    def start_episode(self) -> None:
        """Start a new training episode."""
        self._training_mode = True
        self._current_trajectory = Trajectory([], [], [])

    def end_episode(self) -> float:
        """
        End episode and store trajectory.

        Returns:
            Episode return
        """
        if self._current_trajectory is None:
            return 0.0

        episode_return = self._current_trajectory.total_return
        self._trajectories.append(self._current_trajectory)
        self._episode_returns.append(episode_return)

        self._current_trajectory = None
        self._training_mode = False

        return episode_return

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute control action.

        In training mode: sample from policy and record trajectory.
        In evaluation mode: use policy mean (deterministic).

        Args:
            x: Current state
            t: Current time

        Returns:
            u: Control action
            diagnostics: Policy info
        """
        x = np.asarray(x, dtype=float)

        if self._target is None:
            raise ValueError("Target not set. Call set_target() first.")

        # Compute action
        if self._training_mode:
            action = self._sample_action(x)
        else:
            action = self._policy_mean(x)  # Deterministic

        # Record trajectory if training
        if self._training_mode and self._current_trajectory is not None:
            self._current_trajectory.states.append(x.copy())
            self._current_trajectory.actions.append(action.copy())

        # Error for diagnostics
        error = x - self._target

        # Pseudo-Lyapunov: distance to goal
        V = np.linalg.norm(error) ** 2

        diagnostics = {
            'error': error,
            'error_norm': np.linalg.norm(error),
            'lyapunov': V,
            'policy_mean': self._policy_mean(x),
            'training_mode': self._training_mode,
            'n_episodes': len(self._episode_returns)
        }

        return action, diagnostics

    def record_reward(self, reward: float) -> None:
        """
        Record reward for current step.

        Args:
            reward: Step reward
        """
        if self._training_mode and self._current_trajectory is not None:
            self._current_trajectory.rewards.append(reward)

    def update_policy(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update policy using collected trajectories (REINFORCE).

        Args:
            batch_size: Number of recent trajectories to use (None = all)

        Returns:
            Training statistics
        """
        if not self._trajectories:
            return {'updated': False, 'reason': 'No trajectories'}

        # Select trajectories
        if batch_size is not None:
            trajectories = self._trajectories[-batch_size:]
        else:
            trajectories = self._trajectories

        # Compute policy gradient
        total_gradient = np.zeros_like(self.theta)
        total_return = 0.0

        for traj in trajectories:
            # Compute returns G_t for each timestep
            T = traj.length
            returns = np.zeros(T)
            G = 0.0
            for t in reversed(range(T)):
                G = traj.rewards[t] + self.gamma * G
                returns[t] = G

            # Normalize returns (baseline subtraction)
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

            # Accumulate gradient
            for t in range(T):
                grad = self._log_policy_gradient(traj.states[t], traj.actions[t])
                total_gradient += grad * returns[t]

            total_return += traj.total_return

        # Average over trajectories
        total_gradient /= len(trajectories)

        # Update parameters
        self.theta += self.learning_rate * total_gradient

        # Clear old trajectories (keep recent for monitoring)
        if len(self._trajectories) > 100:
            self._trajectories = self._trajectories[-50:]

        return {
            'updated': True,
            'n_trajectories': len(trajectories),
            'mean_return': total_return / len(trajectories),
            'gradient_norm': np.linalg.norm(total_gradient)
        }

    def train(self, model, n_episodes: int = 100,
              max_steps: int = 100, verbose: bool = True) -> List[float]:
        """
        Train policy on given model.

        Args:
            model: Environment model
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress

        Returns:
            List of episode returns
        """
        self.set_model(model)
        returns = []

        for ep in range(n_episodes):
            # Random initial state
            x0 = np.random.uniform(
                model.x_bounds[:, 0] * 0.8,
                model.x_bounds[:, 1] * 0.8
            )

            self.start_episode()
            x = x0.copy()

            for step in range(max_steps):
                # Get action
                u, _ = self.compute_control(x, step * model.tau)
                u = model.saturate_input(u)

                # Step environment
                w = model.sample_disturbance()
                x_next = model.dynamics(x, u, w)

                # Compute and record reward
                reward = self.compute_reward(x, u, x_next)
                self.record_reward(reward)

                x = x_next

                # Check if goal reached
                if np.linalg.norm(x - self._target) < 0.3:
                    break

            episode_return = self.end_episode()
            returns.append(episode_return)

            # Update policy every few episodes
            if (ep + 1) % 5 == 0:
                stats = self.update_policy(batch_size=10)
                if verbose and (ep + 1) % 20 == 0:
                    print(f"Episode {ep+1}/{n_episodes}: "
                          f"Return={episode_return:.2f}, "
                          f"Mean={np.mean(returns[-20:]):.2f}")

        return returns

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Pseudo-Lyapunov function (distance to goal).

        Note: RL does not provide formal Lyapunov guarantees.

        Args:
            x: State

        Returns:
            ||x - target||^2
        """
        if self._target is None:
            return None
        return float(np.linalg.norm(x - self._target) ** 2)

    def save_policy(self, filepath: str) -> None:
        """Save policy parameters to file."""
        np.savez(filepath, theta=self.theta, sigma=self.sigma)

    def load_policy(self, filepath: str) -> None:
        """Load policy parameters from file."""
        data = np.load(filepath)
        self.theta = data['theta']
        self.sigma = float(data['sigma'])

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self._episode_returns:
            return {'trained': False}

        returns = np.array(self._episode_returns)
        return {
            'trained': True,
            'n_episodes': len(returns),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'max_return': np.max(returns),
            'final_mean_return': np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns),
            'policy_params': self.theta.copy()
        }
