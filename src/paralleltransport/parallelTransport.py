import numpy as np
from vet.vector_operators import tangentPlane, unitVector, surfProjection, convert3Drealto2D

# Transporte paralelo
def transportVector(vertices, basis_x, basis_y, normal, initial_vector = None, connectivity = None, rings = None):
    num_vertices = len(vertices)

    # Plano tangente
    P = tangentPlane(num_vertices, basis_x, basis_y, normal)

    # Vetor unitário
    V = unitVector(vertices, initial_vector)

    # Rotação inicial
    R = np.zeros((num_vertices,3,3))
    R[0] = np.eye(3)

    tol = 1e-6
    if not connectivity:
        # Transporte paralelo por Problema ortogonal de Procrustes
        for i in range(num_vertices-1):
            diff = np.linalg.norm(P[i+1] - P[i])

            if diff < tol:
                R[i+1] = R[i]
                V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])
                continue

            U,S,Vt = np.linalg.svd(P[i].T @ P[i+1])
            R[i+1] = R[i] @ (U @ Vt)
            V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])

    else:
        # Transporte paralelo por Problema ortogonal de Procrustes com conectividade
        visited = np.zeros(num_vertices, dtype=bool)
        for i in range(num_vertices-1):

            diff = np.linalg.norm(P[i+1] - P[i])
            if diff < tol:
                R[i+1] = R[i]
                V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])
                continue

            U,S,Vt = np.linalg.svd(P[i].T @ P[i+1])
            R[i+1] = R[i] @ (U @ Vt)
            V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])

            # Calcula o transporte sequencial sem recálculo dos vértices já visitados
            neighbors = rings[i]
            for j in neighbors:
                if not visited[j]:
                    
                    visited[j] = True 
                    diff = np.linalg.norm(P[j] - P[i])
                    if diff < tol:
                        R[j] = R[i]
                        V[j] = V[i] + R[i] @ P[i].T @ (vertices[j] - vertices[i])
                        continue

                    U_neighbor,S_neighbor,Vt_neighbor = np.linalg.svd(P[i].T @ P[j])
                    R[j] = R[i] @ (U_neighbor @ Vt_neighbor)
                    V[j] = V[i] + R[i] @ P[i].T @ (vertices[j] - vertices[i])
            
            visited[i] = True
    
    # Projetando V na superfície
    V = surfProjection(num_vertices, V, normal, preserve_magnitude=False)

    return R, V

# Transporte paralelo
def transportVectorv2(vertices, basis_x, basis_y, normal, initial_vector = None, connectivity = None, rings = None):
    num_vertices = len(vertices)

    # Plano tangente
    P = tangentPlane(num_vertices, basis_x, basis_y, normal)

    # Vetor unitário
    V = unitVector(vertices, initial_vector)

    # Rotação inicial
    R = np.zeros((num_vertices,3,3))
    R[0] = np.eye(3)

    tol = 1e-6
    if not connectivity:
        # Transporte paralelo por Problema ortogonal de Procrustes
        for i in range(num_vertices-1):
            diff = np.linalg.norm(P[i+1] - P[i])

            if diff < tol:
                R[i+1] = R[i]
                V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])
                continue

            U,S,Vt = np.linalg.svd(P[i].T @ P[i+1])
            R[i+1] = R[i] @ (U @ Vt)
            V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])

    else:
        # Transporte paralelo por Problema ortogonal de Procrustes com conectividade
        visited = np.zeros(num_vertices, dtype=bool)
        for i in range(num_vertices-1):

            diff = np.linalg.norm(P[i+1] - P[i])
            if diff < tol:
                R[i+1] = R[i]
                V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])
                continue

            U,S,Vt = np.linalg.svd(P[i].T @ P[i+1])
            R[i+1] = R[i] @ (U @ Vt)
            V[i+1] = V[i] + R[i] @ P[i].T @ (vertices[i+1] - vertices[i])

            # Calcula o transporte sequencial sem recálculo dos vértices já visitados
            neighbors = rings[i]
            for j in neighbors:
                if not visited[j]:
                    
                    visited[j] = True 
                    diff = np.linalg.norm(P[j] - P[i])
                    if diff < tol:
                        R[j] = R[i]
                        V[j] = V[i] + R[i] @ P[i].T @ (vertices[j] - vertices[i])
                        continue

                    U_neighbor,S_neighbor,Vt_neighbor = np.linalg.svd(P[i].T @ P[j])
                    R[j] = R[i] @ (U_neighbor @ Vt_neighbor)
                    V[j] = V[i] + R[i] @ P[i].T @ (vertices[j] - vertices[i])
            
            visited[i] = True
    
    # Projetando V na superfície
    V = surfProjection(num_vertices, V, normal, preserve_magnitude=False)

    rot = convert3Drealto2D(basis_x,basis_y,V)

    return rot