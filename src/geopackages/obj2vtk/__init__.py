"""
OBJ to VTK Conversion Package

Utilities for converting OBJ mesh files to VTK format, including support
for vector field data and parallel transport visualization.

Functions:
- obj2vtk: Convert OBJ files to VTK format
- addVelocities_obj2vtk: Add velocity vectors to OBJ files and convert to VTK
"""

from .objtovtk import (obj2vtk,
                       addVelocities_obj2vtk
)

__all__ = [
    'obj2vtk',
    'addVelocities_obj2vtk'
]