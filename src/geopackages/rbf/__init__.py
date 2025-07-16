"""
Radial Basis Function Finite Difference Package

Tools for computing surface differential operators using RBF-based finite
difference methods. Supports both 2D and 3D surface operators with reliability
scoring for neighbor selection.

Functions:
- Weight computation: rbf_fd_weights, rbf_fd_weights_derivatives_only
- Surface operators: compute_surface_operators2d, compute_surface_operators3d
- Reliability scoring: compute_surface_operators_score, compute_surface_operators_with_reliability
- Enhanced methods: compute_surface_operators3dd
"""

from .weights_rbf_fd_2d import (rbf_fd_weights,
                                rbf_fd_weights_derivatives_only
)

from .rbf_fd_operators import (compute_surface_operators2d,
                                  compute_surface_operators3d,
                                  compute_surface_operators,
                                  compute_surface_operators_score,
                                  compute_surface_operators_with_reliability,
                                  compute_surface_operators3dd
)

__all__ = [
    'rbf_fd_weights',
    'rbf_fd_weights_derivatives_only',
    'compute_surface_operators2d',
    'compute_surface_operators3d',
    'compute_surface_operators',
    'compute_surface_operators_score',
    'compute_surface_operators_with_reliability',
    'compute_surface_operators3dd'
]
