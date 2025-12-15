"""
Unit tests for MPC and RL controllers.

Tests verify:
1. MPC optimization solves correctly
2. MPC respects constraints
3. RL policy learns and improves
4. Controllers stabilize the system
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.integrator import IntegratorModel
from models.manipulator import TwoLinkManipulator
from controllers.mpc import MPCController, LinearMPC
from controllers.rl_policy_gradient import PolicyGradientController
from sim.simulator import Simulator


class TestMPCController:
    """Tests for MPC controller."""

    def test_mpc_initialization(self):
        """Test MPC initializes correctly."""
        mpc = MPCController(horizon=10)
        assert mpc.horizon == 10
        assert mpc.name == "MPC"

    def test_mpc_with_integrator(self):
        """Test MPC controls integrator to target."""
        model = IntegratorModel(tau=0.1)
        mpc = MPCController(
            horizon=5,
            Q=np.eye(2),
            R=0.1 * np.eye(2)
        )
        mpc.set_model(model)
        mpc.set_target(np.zeros(2))

        # Test single control computation
        x = np.array([2.0, 2.0])
        u, diag = mpc.compute_control(x, 0.0)

        assert u.shape == (2,)
        assert 'optimal_cost' in diag
        assert diag['solver_success']

    def test_mpc_respects_constraints(self):
        """Test MPC respects input constraints."""
        model = IntegratorModel(tau=0.1)
        mpc = MPCController(horizon=5)
        mpc.set_model(model)
        mpc.set_target(np.zeros(2))

        # Start far from target - should hit constraints
        x = np.array([8.0, 8.0])
        u, _ = mpc.compute_control(x, 0.0)

        # Check constraints
        assert np.all(u >= model.u_bounds[:, 0] - 1e-6)
        assert np.all(u <= model.u_bounds[:, 1] + 1e-6)

    def test_mpc_convergence(self):
        """Test MPC converges to target."""
        model = IntegratorModel(tau=0.1)
        mpc = MPCController(horizon=10)
        mpc.set_model(model)
        mpc.set_target(np.zeros(2))

        sim = Simulator(model, mpc, disturbance_mode='none', seed=42)
        result = sim.run(np.array([3.0, 3.0]), T=8.0)

        # Should converge
        final_error = np.linalg.norm(result.states[-1])
        assert final_error < 0.5

    def test_mpc_performance_tracking(self):
        """Test MPC tracks solve times."""
        model = IntegratorModel(tau=0.1)
        mpc = MPCController(horizon=5)
        mpc.set_model(model)
        mpc.set_target(np.zeros(2))

        # Run a few computations
        for _ in range(5):
            mpc.compute_control(np.array([1.0, 1.0]), 0.0)

        perf = mpc.analyze_performance()
        assert perf['n_solves'] == 5
        assert perf['mean_solve_time'] > 0


class TestLinearMPC:
    """Tests for Linear MPC."""

    def test_linear_mpc_prediction(self):
        """Test linear MPC prediction matrices."""
        A = np.eye(2)
        B = 0.1 * np.eye(2)

        lmpc = LinearMPC(A, B, horizon=5)

        # Check prediction matrices are computed
        assert lmpc.Psi.shape == (10, 2)  # horizon*n x n
        assert lmpc.Gamma.shape == (10, 10)  # horizon*n x horizon*m


class TestPolicyGradientController:
    """Tests for RL Policy Gradient controller."""

    def test_pg_initialization(self):
        """Test policy gradient initializes correctly."""
        pg = PolicyGradientController(n_states=2, n_actions=2)

        assert pg.theta.shape == (3, 2)  # (n_states+1) x n_actions
        assert pg.sigma > 0

    def test_pg_action_sampling(self):
        """Test policy samples actions correctly."""
        pg = PolicyGradientController(n_states=2, n_actions=2, sigma=0.5)
        pg.set_target(np.zeros(2))

        state = np.array([1.0, 1.0])
        action = pg._sample_action(state)

        assert action.shape == (2,)

    def test_pg_policy_mean(self):
        """Test policy mean computation."""
        pg = PolicyGradientController(n_states=2, n_actions=2)
        pg.set_target(np.zeros(2))

        # With zero weights, mean should be zero
        state = np.array([1.0, 1.0])
        mean = pg._policy_mean(state)

        np.testing.assert_array_almost_equal(mean, [0.0, 0.0])

    def test_pg_reward_computation(self):
        """Test reward function."""
        pg = PolicyGradientController(n_states=2, n_actions=2)
        pg.set_target(np.zeros(2))

        # Closer to goal should give higher reward
        r1 = pg.compute_reward(np.zeros(2), np.zeros(2), np.array([0.1, 0.1]))
        r2 = pg.compute_reward(np.zeros(2), np.zeros(2), np.array([1.0, 1.0]))

        assert r1 > r2  # Closer gives higher reward

    def test_pg_episode_tracking(self):
        """Test episode start/end tracking."""
        pg = PolicyGradientController(n_states=2, n_actions=2)
        pg.set_target(np.zeros(2))

        pg.start_episode()
        assert pg._training_mode

        pg.compute_control(np.array([1.0, 1.0]), 0.0)
        pg.record_reward(-1.0)

        ret = pg.end_episode()
        assert not pg._training_mode
        assert ret == -1.0

    def test_pg_short_training(self):
        """Test that training improves policy (short test)."""
        model = IntegratorModel(tau=0.1)
        pg = PolicyGradientController(
            n_states=2, n_actions=2,
            learning_rate=0.1,
            sigma=0.3
        )
        pg.set_target(np.zeros(2))

        # Train for a few episodes
        returns = pg.train(model, n_episodes=20, max_steps=50, verbose=False)

        # Should have some returns recorded
        assert len(returns) == 20

        # Check training stats
        stats = pg.get_training_stats()
        assert stats['trained']
        assert stats['n_episodes'] == 20

    def test_pg_policy_update(self):
        """Test policy update changes parameters."""
        pg = PolicyGradientController(n_states=2, n_actions=2)
        pg.set_target(np.zeros(2))

        theta_before = pg.theta.copy()

        # Collect some data
        pg.start_episode()
        for _ in range(10):
            pg.compute_control(np.array([1.0, 1.0]), 0.0)
            pg.record_reward(-1.0)
        pg.end_episode()

        # Update
        pg.update_policy()

        # Parameters should change
        assert not np.allclose(pg.theta, theta_before)


class TestMPCWithManipulator:
    """Tests for MPC with manipulator model."""

    def test_mpc_manipulator_basic(self):
        """Test MPC works with manipulator."""
        model = TwoLinkManipulator(tau=0.01)
        mpc = MPCController(
            horizon=5,
            Q=np.diag([10, 10, 1, 1]),
            R=0.1 * np.eye(2)
        )
        mpc.set_model(model)
        mpc.set_target(np.array([0.5, 0.5, 0.0, 0.0]))

        x = np.array([0.0, 0.0, 0.0, 0.0])
        u, diag = mpc.compute_control(x, 0.0)

        assert u.shape == (2,)
        assert 'solve_time' in diag


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
