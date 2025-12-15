"""
Unit tests for integrator model and controllers.

Tests verify:
1. Model dynamics correctness
2. Controller stability
3. Lyapunov function decrease
4. Safety constraint satisfaction
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.integrator import IntegratorModel
from controllers.proportional import ProportionalController
from controllers.lqr import LQRController
from sim.simulator import Simulator


class TestIntegratorModel:
    """Tests for integrator model."""

    def test_dynamics_zero_input(self):
        """Test dynamics with zero input."""
        model = IntegratorModel(tau=0.1)
        x = np.array([1.0, 2.0])
        u = np.array([0.0, 0.0])

        x_next = model.dynamics(x, u)

        np.testing.assert_array_almost_equal(x_next, x)

    def test_dynamics_unit_input(self):
        """Test dynamics with unit input."""
        model = IntegratorModel(tau=0.1)
        x = np.array([0.0, 0.0])
        u = np.array([1.0, 1.0])

        x_next = model.dynamics(x, u)

        expected = np.array([0.1, 0.1])
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_input_saturation(self):
        """Test input saturation respects bounds."""
        model = IntegratorModel()
        u = np.array([2.0, -2.0])

        u_sat = model.saturate_input(u)

        np.testing.assert_array_almost_equal(u_sat, [1.0, -1.0])

    def test_linearization(self):
        """Test linearized system matrices."""
        model = IntegratorModel(tau=0.1)
        A, B = model.get_linearization(np.zeros(2), np.zeros(2))

        expected_A = np.eye(2)
        expected_B = 0.1 * np.eye(2)

        np.testing.assert_array_almost_equal(A, expected_A)
        np.testing.assert_array_almost_equal(B, expected_B)


class TestProportionalController:
    """Tests for proportional controller."""

    def test_control_at_target(self):
        """Test zero control at target."""
        controller = ProportionalController(kp=0.5)
        controller.set_target(np.array([0.0, 0.0]))

        u, diag = controller.compute_control(np.array([0.0, 0.0]), 0.0)

        np.testing.assert_array_almost_equal(u, [0.0, 0.0])

    def test_control_away_from_target(self):
        """Test control drives toward target."""
        controller = ProportionalController(kp=0.5)
        controller.set_target(np.array([0.0, 0.0]))

        u, diag = controller.compute_control(np.array([1.0, 1.0]), 0.0)

        # Should drive toward origin
        assert u[0] < 0  # Negative control to reduce positive error
        assert u[1] < 0

    def test_stability_analysis(self):
        """Test stability analysis."""
        model = IntegratorModel(tau=0.1)
        controller = ProportionalController(kp=0.5)
        controller.set_model(model)
        controller.set_target(np.zeros(2))

        analysis = controller.analyze_stability(tau=0.1)

        assert analysis['is_stable']
        assert analysis['spectral_radius'] < 1.0


class TestLQRController:
    """Tests for LQR controller."""

    def test_design(self):
        """Test LQR design produces valid gain."""
        model = IntegratorModel(tau=0.1)
        controller = LQRController()

        K = controller.design_for_model(model)

        assert K is not None
        assert K.shape == (2, 2)

    def test_control_converges(self):
        """Test LQR control converges to target."""
        model = IntegratorModel(tau=0.1)
        controller = LQRController()
        controller.design_for_model(model)
        controller.set_target(np.array([0.0, 0.0]))

        sim = Simulator(model, controller, disturbance_mode='none', seed=42)
        result = sim.run(np.array([5.0, 5.0]), T=10.0)

        # Should converge close to target
        final_error = np.linalg.norm(result.states[-1])
        assert final_error < 0.1

    def test_lyapunov_decrease(self):
        """Test Lyapunov function decreases along trajectory."""
        model = IntegratorModel(tau=0.1)
        controller = LQRController()
        controller.design_for_model(model)
        controller.set_target(np.array([0.0, 0.0]))

        sim = Simulator(model, controller, disturbance_mode='none', seed=42)
        result = sim.run(np.array([3.0, 3.0]), T=5.0)

        V = result.lyapunov_values

        # Lyapunov should decrease (monotonically for no disturbance)
        for i in range(len(V) - 1):
            if V[i] > 1e-8:  # Not at equilibrium
                assert V[i + 1] <= V[i] + 1e-8  # Allow small numerical error


class TestSimulator:
    """Tests for simulation engine."""

    def test_simulation_length(self):
        """Test simulation produces correct length trajectory."""
        model = IntegratorModel(tau=0.1)
        controller = ProportionalController(kp=0.5)
        controller.set_target(np.zeros(2))

        sim = Simulator(model, controller, seed=42)
        result = sim.run(np.array([1.0, 1.0]), T=1.0)

        expected_steps = int(1.0 / 0.1) + 1
        assert len(result.time) == expected_steps
        assert result.states.shape[0] == expected_steps

    def test_disturbance_bounds(self):
        """Test disturbances stay within bounds."""
        model = IntegratorModel(tau=0.1)
        controller = ProportionalController(kp=0.5)
        controller.set_target(np.zeros(2))

        sim = Simulator(model, controller, disturbance_mode='random', seed=42)
        result = sim.run(np.array([1.0, 1.0]), T=5.0)

        # All disturbances should be within bounds
        assert np.all(result.disturbances >= -0.05)
        assert np.all(result.disturbances <= 0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
