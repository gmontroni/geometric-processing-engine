"""
Vector Operations and Mesh Data Structures Package

Comprehensive toolkit for 3D vector operations, mesh processing, and geometric
computations. Includes a mesh data structure (VET) and optimized vector operations.

Classes:
- VET: Vertex-Edge-Triangle mesh data structure

Functions:
- Vector operations: cross3d, dot3d, norm3d, normalize
- Geometric projections: surfProjection, tangentPlane
- Coordinate conversions: convert3Drealto2D, convert2Dcomplexto3Dreal
- Complex operations: complexRotation
"""

from .pyvet import VET
from .vector_operators import (cross3d,
                               dot3d,
                               norm3d,
                               normalize,
                               tangentPlane,
                               normalizeVector,
                               vertexRings,
                               unitVector,
                               surfProjection,
                               convert3Drealto2D,
                               convert3Drealto2Dcomplex,
                               rotationMatrixToAngle,
                               convert2Dcomplexto3Dreal,
                               complexRotation
)

__all__ = [
    'VET',
    'cross3d',
    'dot3d',
    'norm3d',
    'normalize',
    'tangentPlane',
    'normalizeVector',
    'vertexRings',
    'unitVector',
    'surfProjection',
    'convert3Drealto2D',
    'convert3Drealto2Dcomplex',
    'rotationMatrixToAngle',
    'convert2Dcomplexto3Dreal',
    'complexRotation'
]