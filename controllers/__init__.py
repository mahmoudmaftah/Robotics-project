"""
Controller implementations for discrete-time robot control.

All controllers implement a standard interface via the Controller base class.
"""

from .base import Controller
from .proportional import ProportionalController
from .lqr import LQRController
from .pid import PIDController
from .mpc import MPCController, LinearMPC
from .rl_policy_gradient import PolicyGradientController

__all__ = [
    'Controller',
    'ProportionalController',
    'LQRController',
    'PIDController',
    'MPCController',
    'LinearMPC',
    'PolicyGradientController'
]
