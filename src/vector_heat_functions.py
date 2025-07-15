import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix

def unit(v):
    return v / np.sqrt(np.sum(v**2))

def normalizeVector(v):
    return v / np.linalg.norm(v)

def buildTangentBasis(normal):
    unit_dir = normalizeVector(normal)
    test_vec = np.array([1., 0., 0.])

    if np.abs(np.dot(test_vec, unit_dir)) > 0.9:
        test_vec = np.array([0., 1., 0.])

    basis_x = normalizeVector(np.cross(test_vec, unit_dir))
    basis_y = normalizeVector(np.cross(unit_dir, basis_x))

    tangent_basis = np.array([basis_x, basis_y])

    return tangent_basis

def buildSourceVectors(num_points, source_idxs, source_vecs):

    source_vectors = np.zeros((num_points, 2))
    for i, idx in enumerate(source_idxs):
        source_vectors[idx] = source_vecs[i]

    sources = list(zip(source_idxs, source_vecs))

    return sources, source_vectors

def angleInPlane(u, v, normal):
    n = unit(normal)
    u_plane = u - np.dot(u, n) * n
    basis_y = unit(np.cross(n, u_plane))

    x_comp = np.dot(v, u_plane)
    y_comp = np.dot(v, basis_y)

    return np.arctan2(y_comp,x_comp)

# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def rotateAround(this_v, axis, theta):
    axis_n = axis / np.linalg.norm(axis)

    # this_v paralelo ao eixo de rotação axis_n
    parallel_comp = axis_n * np.dot(this_v, axis_n)
    # componente perpendicular
    tangent_comp = this_v - parallel_comp

    # base local
    tangent_mag = np.linalg.norm(tangent_comp)
    if tangent_mag > 0:
        basis_x = tangent_comp / tangent_mag
        basis_y = np.cross(axis_n, basis_x)
    
        rotated_vec = tangent_mag * (np.cos(theta) * basis_x + np.sin(theta) * basis_y)
        
        return rotated_vec + parallel_comp
    else:
        return parallel_comp
    
def transportBetweenOriented(p_source, p_target, normals, tangent_basis):

    source_n = normals[p_source]
    source_basis_x = tangent_basis[0][p_source]

    target_n = normals[p_target]
    target_basis_x = tangent_basis[0][p_target]
    target_basis_y = tangent_basis[1][p_target]

    inverted = False
    if np.dot(source_n, target_n) < 0:
        target_n *= -1
        target_basis_y *= -1
        inverted = True

    axis = np.cross(source_n, target_n)
    if np.linalg.norm(axis) > 1e-6:
        axis = unit(axis)
    else:
        axis = source_basis_x

    # Cálculo do ângulo entre os vetores normais (unfold)
    angle = angleInPlane(source_n, target_n, axis)

    # fórmula de rotação de Rodrigues (translate +  fold)
    source_x_in_target3 = rotateAround(source_basis_x, axis, angle)
    source_x_in_target = np.array([np.dot(source_x_in_target3, target_basis_x),np.dot(source_x_in_target3, target_basis_y)])

    return (source_x_in_target, inverted)

def transportBetweenOrientedSVD(p_source, p_target, normals, tangent_basis):
    
    source_n = normals[p_source]
    source_basis_x = tangent_basis[0][p_source]
    source_basis_y = tangent_basis[1][p_source]
    
    target_n = normals[p_target]
    target_basis_x = tangent_basis[0][p_target]
    target_basis_y = tangent_basis[1][p_target]
    
    inverted = False
    if np.dot(source_n, target_n) < 0:
        target_n = -target_n
        target_basis_y = -target_basis_y
        inverted = True
    
    P_source = np.vstack([source_basis_x, source_basis_y, source_n])
    P_target = np.vstack([target_basis_x, target_basis_y, target_n])
    
    U, S, Vt = np.linalg.svd(P_source.T @ P_target)
    R = U @ Vt
    
    transported_basis_x = R @ source_basis_x
    
    source_x_in_target = np.array([np.dot(transported_basis_x, target_basis_x),np.dot(transported_basis_x, target_basis_y)])
    
    return source_x_in_target, inverted

# Versão atualizada
def computeConnectionLaplacian(cloud, laplacian, normals, tangent_basis):
    n = cloud.shape[0]
    # Use listas para construir a matriz esparsa de forma eficiente
    rows, cols, data = [], [], []
    
    def addComplexCoef(i, j, val, conj):
        c = -1. if conj else 1.
        rows.extend([2*i, 2*i, 2*i+1, 2*i+1])
        cols.extend([2*j, 2*j+1, 2*j, 2*j+1])
        data.extend([val[0], -val[1]*c, val[1], val[0]*c])
    
    # Iterar apenas sobre elementos não-zero
    for i, j in zip(*laplacian.nonzero()):
        if i == j:
            continue
            
        this_val = laplacian[i, j]
        r_ij, inverted = transportBetweenOriented(j, i, normals, tangent_basis)
        
        addComplexCoef(i, j, this_val * r_ij, inverted)
        addComplexCoef(i, i, -this_val * np.array([1., 0]), False)
    
    return csc_matrix((data, (rows, cols)), shape=(2*n, 2*n))

def computeConnectionLaplacianSVD(cloud, laplacian, normals, tangent_basis):
    n = cloud.shape[0]
    rows, cols, data = [], [], []
    
    def addComplexCoef(i, j, val, conj):
        c = -1. if conj else 1.
        rows.extend([2*i, 2*i, 2*i+1, 2*i+1])
        cols.extend([2*j, 2*j+1, 2*j, 2*j+1])
        data.extend([val[0], -val[1]*c, val[1], val[0]*c])
    
    for i, j in zip(*laplacian.nonzero()):
        if i == j:
            continue
            
        this_val = laplacian[i, j]
        r_ij, inverted = transportBetweenOrientedSVD(j, i, normals, tangent_basis)
        
        addComplexCoef(i, j, this_val * r_ij, inverted)
        addComplexCoef(i, i, -this_val * np.array([1., 0]), False)
    
    return csc_matrix((data, (rows, cols)), shape=(2*n, 2*n))

def complexToReal(m):
    n_row, n_col = m.shape
    data, row, col = [], [], []

    matrix = csr_matrix(m)
    matrix = matrix.tocoo()  # Converte para formato COO para iteração eficiente
    
    for i, j, val in zip(matrix.row, matrix.col, matrix.data):
        data.extend([val.real, -val.imag, val.imag, val.real])
        row.extend([2 * i, 2 * i, 2 * i + 1, 2 * i + 1])
        col.extend([2 * j, 2 * j + 1, 2 * j, 2 * j + 1])
    
    real_matrix = coo_matrix((data, (row, col)), shape=(2 * n_row, 2 * n_col))
    return real_matrix.tocsc()  # Retorna no formato CSC para eficiência

def realtoComplex(vec):
    n_vec = len(vec) // 2  
    complex_vec = np.zeros(n_vec, dtype=np.complex128)  
    
    for i in range(n_vec):
        complex_vec[i] = vec[2 * i] + 1j * vec[2 * i + 1]
    
    return complex_vec

def complextoReal(vec):
    n_vec = len(vec)
    real_vec = np.zeros(2 * n_vec, dtype=np.float64)

    for i in range(n_vec):
        real_vec[2 * i] = vec[i].real
        real_vec[2 * i + 1] = vec[i].imag  

    return real_vec

def buildY0(sources, pts_count):
    dir_rhs = np.zeros(pts_count, dtype=np.complex128)

    norms_all_same = True
    if len(sources) > 0:
        first_norm = np.linalg.norm(sources[0][1])
        
        for pt_idx, vec in sources:
            dir_rhs[pt_idx] += complex(vec[0], vec[1])
            
            # Verifica se a norma é igual à do primeiro vetor
            this_norm = np.linalg.norm(vec)
            if abs(first_norm - this_norm) > max(first_norm, this_norm) * 1e-10:
                norms_all_same = False
    
    return dir_rhs, norms_all_same, first_norm

def buildVecHeatOperator(mass_matrix, connection_laplacian_real, t):

    mass_real = complexToReal(mass_matrix.astype(np.complex128))
    operator = mass_real - t * connection_laplacian_real

    return operator

def solveVectorHeat(operator, y0_complex):

    y0_real = complextoReal(y0_complex)
    result_real = spsolve(operator, y0_real)
    
    return result_real