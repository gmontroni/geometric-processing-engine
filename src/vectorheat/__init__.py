"""
Vector Heat Method Package

Implementation of the vector heat method for computing vector fields on surfaces.
Provides tools for solving vector diffusion problems and computing connection
Laplacians with parallel transport.

Functions:
- Vector field processing: buildTangentBasis, buildSourceVectors
- Parallel transport: transportBetweenOriented, transportBetweenOrientedSVD
- Connection Laplacian: computeConnectionLaplacian, computeConnectionLaplacianSVD
- Heat equation solving: buildVecHeatOperator, solveVectorHeat
- Complex/real conversions: complexToReal, realtoComplex
"""

from .vector_heat_functions import (unit,
                                    normalizeVector,
                                    buildTangentBasis,
                                    buildSourceVectors,
                                    angleInPlane,
                                    rotateAround,
                                    transportBetweenOriented,
                                    transportBetweenOrientedSVD,
                                    computeConnectionLaplacian,
                                    computeConnectionLaplacianSVD,
                                    complexToReal,
                                    realtoComplex,
                                    complextoReal,
                                    buildY0,
                                    buildVecHeatOperator,
                                    solveVectorHeat
)

__all__ = [
    'unit',
    'normalizeVector',
    'buildTangentBasis',
    'buildSourceVectors',
    'angleInPlane',
    'rotateAround',
    'transportBetweenOriented',
    'transportBetweenOrientedSVD',
    'computeConnectionLaplacian',
    'computeConnectionLaplacianSVD',
    'complexToReal',
    'realtoComplex',
    'complextoReal',
    'buildY0',
    'buildVecHeatOperator',
    'solveVectorHeat'
]