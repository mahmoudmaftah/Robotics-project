"""
Two-link manipulator controllers.

Controllers for the planar two-link robot arm.
"""

from .computed_torque import ComputedTorqueController
from .pd_gravity import PDGravityCompensation
from .backstepping import BacksteppingController
from .manipulator_lqr import ManipulatorLQRController

__all__ = [
    'ComputedTorqueController',
    'PDGravityCompensation',
    'BacksteppingController',
    'ManipulatorLQRController'
]
