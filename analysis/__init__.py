"""
Stability analysis module.

Provides Lyapunov-based stability certificates and numerical verification.
"""

from .lyapunov import LyapunovAnalyzer, verify_lyapunov_decrease

__all__ = ['LyapunovAnalyzer', 'verify_lyapunov_decrease']
