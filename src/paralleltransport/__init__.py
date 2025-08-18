"""
Parallel Transport Package

Implements parallel transport of vectors along surfaces using orthogonal
Procrustes problem solutions. Supports both sequential and connectivity-based
transport methods for maintaining vector orientations across surface meshes.

Functions:
- transportVector: Basic parallel transport with optional connectivity
- transportVectorv2: Enhanced version with 2D coordinate conversion
"""

from .parallelTransport import (transportVector,
                                transportVectorv2
)

__all__ = [
    'transportVector',
    'transportVectorv2' 
]