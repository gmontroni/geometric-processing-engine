import numpy as np
import numpy.matlib
from numba import njit, prange

def cross3d(u, v):
    return np.array([u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]])

def dot3d(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def norm3d(u):
    return np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])

# normalize a X with shape (n,3)
def normalize(X):
    L = np.sqrt(np.sum(X * X, 1))
    return X / numpy.matlib.repmat(L.reshape((L.shape[0], 1)), 1, 3)

def tangentPlane(num_vertices, basis_x, basis_y, basis_z):

    tangent_plane = np.zeros((num_vertices,3,3))
    for i in range(num_vertices):
        tangent_plane[i] = np.vstack([basis_z[i],basis_x[i],basis_y[i]])
        # P[i] = np.column_stack([basis_z[i],basis_x[i],basis_y[i]])
    return tangent_plane

def normalizeVector(v):
    return v/norm3d(v)

def vertexRings(mesh, num_vertices):

    rings = []
    for i in range(num_vertices):
        rings.append(mesh.compute_ring(i))

    return rings

def unitVector(vertices, initial_vector):
    num_vertices = len(vertices)

    V = np.zeros((num_vertices,3))
    if initial_vector is None:
        V[0] = normalizeVector(vertices[0])
    else:
        V[0] = np.array(initial_vector)  # Removida a normalização
        # V[0] = normalizeVector(np.array(initialVector))
    return V

def surfProjection(num_vertices, vec, normal, preserve_magnitude=True):
    for i in range(num_vertices):
        if not np.isnan(vec[i]).any() and not np.isinf(vec[i]).any():
            
            original_magnitude = norm3d(vec[i])
            vec[i] = vec[i] - dot3d(vec[i],normal[i]) * normal[i]
            
            if preserve_magnitude:
                # Restaurar à magnitude original
                current_magnitude = norm3d(vec[i])
                if current_magnitude > 1e-10:  # Evitar divisão por quase-zero
                    vec[i] = vec[i] * (original_magnitude / current_magnitude)
            else:
                vec[i] = normalizeVector(vec[i])

    return vec

def convert3Drealto2D(axis_x, axis_y, v):
    num_vertices = len(v)

    v2d = np.zeros((num_vertices,2))
    for i in range(num_vertices):
        x_comp = np.dot(v[i],axis_x[i])
        y_comp = np.dot(v[i],axis_y[i])
        v2d[i] = np.array([x_comp, y_comp])

    return v2d

def convert3Drealto2Dcomplex(axis_x, axis_y, v):
    num_vertices = len(v)

    # z = np.zeros((num_vertices,2))
    z = np.zeros((num_vertices),dtype=np.complex128)
    angle = np.zeros(num_vertices)
    for i in range(num_vertices):
        # z[i,0] = np.dot(v[i],axis_x[i])
        # z[i,1] = np.dot(v[i],axis_y[i])
        # angle[i] = np.arctan2(z[i,1],z[i,0])

        a = np.dot(v[i],axis_x[i])
        b = np.dot(v[i],axis_y[i])
        z[i] = complex(a,b)
        angle[i] = np.arctan2(b,a)

    return z, angle

def rotationMatrixToAngle(rot_matrix):
    cos_theta = (np.trace(rot_matrix) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)

    return np.array([np.cos(angle), np.sin(angle)]).shape

def convert2Dcomplexto3Dreal(axis_x, axis_y, z):
    num_vertices = len(z)

    v = np.zeros((num_vertices,3))
    for i in range(num_vertices):
        v[i] = z[i].real * axis_x[i] + z[i].imag * axis_y[i]

    normalized_v = normalize(v)

    return normalized_v

@njit(parallel=True)
def complexRotation(angle):
    num_vertices = len(angle)
    R = np.zeros((num_vertices, num_vertices), dtype=np.complex128)

    # prange para paralelizar o loop
    for i in prange(num_vertices):
        for j in range(i, num_vertices):
            angle_diff = np.abs(angle[i] - angle[j])
            R[i, j] = np.cos(angle_diff) + 1j * np.sin(angle_diff)
    
    # prange para paralelizar o loop
    for i in prange(1, num_vertices):
        for j in range(i):
            R[i, j] = np.conj(R[j, i])
    return R