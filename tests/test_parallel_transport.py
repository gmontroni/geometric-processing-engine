import numpy as np
import pytest
import meshio, sys, os

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from pyvet import VET
from parallelTransport import parallelTransport
from vector_operators import tangentPlane

def test_parallel_transport_output_shape():
    """Test that the function returns outputs of expected shapes."""
    # Create a simple test mesh (3 vertices in a triangle)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    T = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])
    B = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    N = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])
    initial_vector = [1, 0, 0]
    
    R, V = parallelTransport(vertices, T, B, N, initial_vector)
    
    # Check output shapes
    assert R.shape == (3, 3, 3)  # num_vertices x 3 x 3 rotation matrices
    assert V.shape == (3, 3)    # num_vertices x 3 vectors

# def test_vector_magnitude_preservation():
#     """Test that the magnitude of the transported vector is preserved."""
#     # # Create a simple test mesh
#     # vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
#     # T = np.array([
#     #     [1, 0, 0],
#     #     [1, 0, 0],
#     #     [1, 0, 0],
#     #     [1, 0, 0]
#     # ])
#     # B = np.array([
#     #     [0, 1, 0],
#     #     [0, 1, 0],
#     #     [0, 1, 0],
#     #     [0, 1, 0]
#     # ])
#     # N = np.array([
#     #     [0, 0, 1],
#     #     [0, 0, 1],
#     #     [0, 0, 1],
#     #     [0, 0, 1]
#     # ])
#     # Input mesh
#     fileName = 'input/mesh.obj'
#     mesh = meshio.read(fileName, file_format='obj')
#     pts = mesh.points
#     tri = np.array(mesh.cells_dict['triangle'])
#     nopts, _ = pts.shape[0], tri.shape[0]

#     # Source
#     source = 0

#     # Construção das coordenadas locais
#     mesh = VET(pts, tri)
#     # rings = vertexRings(mesh, pts.shape[0])
#     T, B, N = mesh.computeOrthonormalBase()   

#     initial_vector = [1, 1, 0]  # Initial vector in tangent plane
    
#     _, V = parallelTransport(pts, T, B, N, initial_vector)
    
#     # Check vector magnitude preservation
#     initial_norm = np.linalg.norm(initial_vector)
#     for i in range(len(pts)):
#         assert np.isclose(np.linalg.norm(V[i]), initial_norm, atol=1e-5)

def test_vectors_in_tangent_plane():
    """Test that transported vectors stay in the tangent plane (orthogonal to normal)."""
    # Create a simple curved surface (part of a sphere)
    vertices = np.array([
        [0, 0, 1],
        [0.5, 0, 0.866],
        [0, 0.5, 0.866]
    ])
    
    # Custom tangent, binormal, and normal vectors
    T = np.array([
        [1, 0, 0],
        [0.866, 0, -0.5],
        [0.866, 0, -0.5]
    ])
    B = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [-0.5, 0.866, 0]
    ])
    N = np.array([
        [0, 0, 1],
        [0.5, 0, 0.866],
        [0, 0.5, 0.866]
    ])
    initial_vector = [1, 0, 0]
    
    _, V = parallelTransport(vertices, T, B, N, initial_vector)
    
    # Check if vectors are perpendicular to normals (dot product close to zero)
    for i in range(len(vertices)):
        dot_product = np.dot(V[i], N[i])
        assert np.isclose(dot_product, 0, atol=1e-4)

def test_flat_surface_rotation_matrices():
    """Test that on a flat surface, all rotation matrices are identical."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]
    ])
    T = np.tile([1, 0, 0], (5, 1))
    B = np.tile([0, 1, 0], (5, 1))
    N = np.tile([0, 0, 1], (5, 1))
    initial_vector = [1, 1, 0]
    
    R, _ = parallelTransport(vertices, T, B, N, initial_vector)
    
    # Em uma superfície plana, todas as matrizes de rotação devem ser iguais
    for i in range(1, len(vertices)):
        assert np.allclose(R[i], R[0], atol=1e-5)

def test_identical_tangent_planes():
    """Test handling of identical adjacent tangent planes."""
    # Create vertices with identical tangent planes
    vertices = np.array([
        [0, 0, 0],
        [0.001, 0, 0],  # Very close to first vertex
        [1, 0, 0]
    ])
    T = np.tile([1, 0, 0], (3, 1))
    B = np.tile([0, 1, 0], (3, 1))
    N = np.tile([0, 0, 1], (3, 1))
    initial_vector = [1, 0, 0]
    
    _, V = parallelTransport(vertices, T, B, N, initial_vector)
    
    # The transported vector should still be defined for all points
    for i in range(len(vertices)):
        assert not np.any(np.isnan(V[i]))

def test_transport_reversibility():
    """Test that parallel transport is reversible - transporting from A to B and back to A yields the original vector."""
    # Create a simple curved surface (part of a sphere)
    vertices = np.array([
        [0, 0, 1],             # Ponto A (topo da esfera)
        [0.5, 0, 0.866],       # Ponto B (na esfera)
        [0, 0, 1]              # Repetir ponto A para teste de reversibilidade
    ])
    
    # Tangent, binormal, and normal vectors
    T = np.array([
        [1, 0, 0],
        [0.866, 0, -0.5],
        [1, 0, 0]              # Mesmo que no ponto A
    ])
    B = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]              # Mesmo que no ponto A
    ])
    N = np.array([
        [0, 0, 1],
        [0.5, 0, 0.866],
        [0, 0, 1]              # Mesmo que no ponto A
    ])
    
    # Vetor inicial no plano tangente do ponto A
    initial_vector = [0.7, 0.7, 0]  # Vetor no plano XY (tangente ao polo norte)
    
    # Transporte paralelo do ponto A -> B -> A (caminho de ida e volta)
    _, V = parallelTransport(vertices, T, B, N, initial_vector)
    
    # Verificar se o vetor transportado de volta ao ponto A (índice 2) 
    # é igual ao vetor original no ponto A (índice 0)
    assert np.allclose(V[0], V[2], atol=1e-5)
    
    # Verificação alternativa: comparar com o vetor inicial
    initial_vector_array = np.array(initial_vector)
    assert np.allclose(initial_vector_array, V[2], atol=1e-5)

def test_sphere_parallel_transport_holonomy():
    """Test parallel transport around a closed loop on a sphere (should exhibit holonomy/deficit angle)."""
    # Create vertices forming a loop on a sphere (1/8 of the sphere - 45° x 45°)
    theta = np.linspace(0, np.pi/4, 4)  # Ângulo polar (latitude)
    phi = np.linspace(0, np.pi/4, 4)    # Ângulo azimutal (longitude)
    
    # Criar uma trajetória fechada na esfera
    vertices = []
    for t in theta:
        # Ponto em longitude zero
        x = np.sin(t) * np.cos(0)
        y = np.sin(t) * np.sin(0)
        z = np.cos(t)
        vertices.append([x, y, z])
    
    for p in phi[1:]:
        # Manter latitude máxima, variar longitude
        x = np.sin(theta[-1]) * np.cos(p)
        y = np.sin(theta[-1]) * np.sin(p)
        z = np.cos(theta[-1])
        vertices.append([x, y, z])
    
    for t in reversed(theta[:-1]):
        # Voltar pela longitude máxima
        x = np.sin(t) * np.cos(phi[-1])
        y = np.sin(t) * np.sin(phi[-1])
        z = np.cos(t)
        vertices.append([x, y, z])
    
    for p in reversed(phi[:-1]):
        # Voltar para o início
        x = np.sin(theta[0]) * np.cos(p)
        y = np.sin(theta[0]) * np.sin(p)
        z = np.cos(theta[0])
        vertices.append([x, y, z])
    
    # Repetir o primeiro vértice para fechar o loop
    vertices.append(vertices[0])
    vertices = np.array(vertices)
    
    # Calcular bases locais (T, B, N) em cada ponto
    N = vertices.copy()  # Em uma esfera, a normal é o próprio vetor posição normalizado
    for i in range(len(vertices)):
        N[i] = N[i] / np.linalg.norm(N[i])
    T = np.zeros_like(vertices)
    B = np.zeros_like(vertices)
    
    # Calcular T e B para cada vértice
    for i in range(len(vertices)):
        # Escolha um vetor não paralelo a N para calcular um vetor tangente
        if abs(N[i, 2]) < 0.9:  # Se não estamos perto do polo
            aux = np.array([0, 0, 1])
        else:
            aux = np.array([1, 0, 0])
            
        # Primeiro vetor tangente (T) usando produto vetorial e normalização
        T[i] = np.cross(aux, N[i])
        T[i] = T[i] / np.linalg.norm(T[i])
        
        # Segundo vetor tangente (B) completando a base ortonormal
        B[i] = np.cross(N[i], T[i])
        B[i] = B[i] / np.linalg.norm(B[i])
    
    # Vetor inicial tangente à esfera no primeiro ponto
    initial_vector = T[0].copy()
    
    # Realizar transporte paralelo
    R, V = parallelTransport(vertices, T, B, N, initial_vector)
    
    # Após transporte ao redor do loop, o vetor deve retornar com um ângulo de déficit
    # relacionado ao ângulo sólido da região percorrida
    
    # Calcular ângulo entre vetor inicial e vetor final (após loop completo)
    cos_angle = np.dot(V[0], V[-1]) / (np.linalg.norm(V[0]) * np.linalg.norm(V[-1]))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # VERIFICAÇÃO MANUAL: aplicar diretamente R[-1] ao vetor inicial
    manually_rotated = R[-1] @ initial_vector
    cos_angle_manual = np.dot(initial_vector, manually_rotated) / (
        np.linalg.norm(initial_vector) * np.linalg.norm(manually_rotated)
    )
    angle_manual = np.arccos(np.clip(cos_angle_manual, -1.0, 1.0))

    # Ângulo sólido para um octante de esfera é π/2
    expected_deficit = np.pi/2

    print(f"Ângulo original: {angle}")
    print(f"Ângulo calculado manualmente: {angle_manual}")
    print(f"Esperado: {expected_deficit}")

    # Verificar se o ângulo manual está correto
    assert np.isclose(angle_manual, expected_deficit, atol=0.2)

    # Verificar se os vetores do algoritmo correspondem aos vetores manualmente rotacionados
    assert np.allclose(V[-1], manually_rotated, atol=1e-5)

def test_rotation_matrices_orthogonality():
    """Test that all rotation matrices are orthogonal, ensuring preservation of vector magnitudes."""
    # Usar sua malha existente ou uma malha de teste simples
    fileName = 'input/mesh.obj'
    mesh = meshio.read(fileName, file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    
    R, _ = parallelTransport(pts, T, B, N, [1, 1, 0])
    
    # Verificar ortogonalidade: R^T * R ≈ I para cada matriz
    for i in range(len(pts)):
        product = R[i].T @ R[i]
        assert np.allclose(product, np.eye(3), atol=1e-5)
        
        # Verificar também determinante = 1 (preserva orientação)
        assert np.isclose(np.linalg.det(R[i]), 1.0, atol=1e-5)

def test_sphere_rotation_matrices_capture_curvature():
    """Test that rotation matrices correctly capture the curvature of a sphere."""
    # Criar um caminho triangular simples em uma esfera (1/8 da esfera)
    vertices = np.array([
        [0, 0, 1],                  # Polo norte
        [1/np.sqrt(2), 0, 1/np.sqrt(2)],  # Ponto em longitude 0°, latitude 45°
        [1/np.sqrt(2), 1/np.sqrt(2), 0],  # Ponto em longitude 45°, latitude 90°
        [0, 0, 1]                   # Volta ao polo norte para fechar o loop
    ])
    
    # Calcular bases locais (T, B, N) em cada ponto
    N = vertices.copy()  # Normal = vetor posição normalizado
    for i in range(len(vertices)):
        N[i] = N[i] / np.linalg.norm(N[i])
    
    T = np.zeros_like(vertices)
    B = np.zeros_like(vertices)
    
    # Calcular T e B para cada vértice
    for i in range(len(vertices)):
        # Escolha um vetor não paralelo a N
        if abs(N[i, 2]) < 0.9:
            aux = np.array([0, 0, 1])
        else:
            aux = np.array([1, 0, 0])
        
        # Primeiro vetor tangente (T)
        T[i] = np.cross(aux, N[i])
        T[i] = T[i] / np.linalg.norm(T[i])
        
        # Segundo vetor tangente (B)
        B[i] = np.cross(N[i], T[i])
        B[i] = B[i] / np.linalg.norm(B[i])
    
    # Vetor inicial tangente à esfera no primeiro ponto
    initial_vector = T[0].copy()
    
    # Calcular matrizes de rotação e vetores transportados
    R, V = parallelTransport(vertices, T, B, N, initial_vector)
    
    # Verificações Diretas nas Matrizes de Rotação:
    
    # 1. Verificar que as matrizes de rotação são diferentes em diferentes pontos
    assert not np.allclose(R[0], R[1], atol=1e-5), "Rotation matrices should differ on curved surface"
    assert not np.allclose(R[1], R[2], atol=1e-5), "Rotation matrices should differ on curved surface"
    
    # 2. Verificar produto de rotações ao longo do caminho (é isso que captura a curvatura)
    # Calcule rotações ponto a ponto
    R_01 = R[1] @ np.linalg.inv(R[0])  # Rotação do ponto 0 para 1
    R_12 = R[2] @ np.linalg.inv(R[1])  # Rotação do ponto 1 para 2
    R_20 = R[0] @ np.linalg.inv(R[2])  # Rotação do ponto 2 de volta para 0
    
    # O produto R_20 @ R_12 @ R_01 deve capturar o déficit angular
    rotation_loop = R_20 @ R_12 @ R_01
    
    # Extrair ângulo desta matriz de rotação
    # trace(R) = 1 + 2*cos(θ) para rotação de ângulo θ
    cos_angle = (np.trace(rotation_loop) - 1) / 2
    deficit_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Ângulo esperado para um triângulo esférico cobrindo 1/8 da esfera
    expected_deficit = np.pi/2  # 90 graus
    
    print(f"Déficit angular das rotações: {deficit_angle}")
    print(f"Ângulo esperado: {expected_deficit}")
    print(f"Rotação individual R[1]:\n{R[1]}")
    print(f"Produto das rotações ao redor do loop:\n{rotation_loop}")
    
    # Verificar se o déficit angular está próximo do esperado
    assert np.isclose(deficit_angle, expected_deficit, atol=0.2), \
        f"Expected deficit angle {expected_deficit}, got {deficit_angle}"
    
    # Verificar se a aplicação direta da rotação final gera o mesmo resultado
    assert np.allclose(V[-1], R[-1] @ initial_vector, atol=1e-5), \
        "Vector transport should match direct application of rotation matrix"

def test_orthonormal_bases_and_orientation_preservation():
    """Test that tangent bases are orthonormal and rotations preserve orientation."""
    # Input mesh
    fileName = 'input/bunny.obj'
    mesh = meshio.read(fileName, file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0]

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    # rings = vertexRings(mesh, pts.shape[0])
    T, B, N = mesh.computeOrthonormalBase()   
    
    # TESTE 1: Verificar se T, B, N são unitários e ortogonais
    for i in range(len(pts)):
        # Verificar se são unitários
        assert np.isclose(np.linalg.norm(T[i]), 1.0, atol=1e-5), f"T não é unitário no ponto {i}"
        assert np.isclose(np.linalg.norm(B[i]), 1.0, atol=1e-5), f"B não é unitário no ponto {i}"
        assert np.isclose(np.linalg.norm(N[i]), 1.0, atol=1e-5), f"N não é unitário no ponto {i}"
        
        # Verificar ortogonalidade (produto escalar deve ser zero)
        assert np.isclose(np.dot(T[i], B[i]), 0.0, atol=1e-5), f"T e B não são ortogonais no ponto {i}"
        assert np.isclose(np.dot(T[i], N[i]), 0.0, atol=1e-5), f"T e N não são ortogonais no ponto {i}"
        assert np.isclose(np.dot(B[i], N[i]), 0.0, atol=1e-5), f"B e N não são ortogonais no ponto {i}"
        
        # Verificar orientação correta (T × B = N)
        cross_product = np.cross(T[i], B[i])
        assert np.allclose(cross_product, N[i], atol=1e-5), f"T×B ≠ N no ponto {i}"
    
    # Realizar transporte paralelo e obter as matrizes de rotação
    initial_vector = T[0].copy()
    R, _ = parallelTransport(pts, T, B, N, initial_vector)
    
    # TESTE 2: Verificar preservação de orientação nas matrizes de rotação
    for i in range(len(pts)):
        # Verificar se det(R) = 1
        det = np.linalg.det(R[i])
        assert np.isclose(det, 1.0, atol=1e-5), f"Determinante de R[{i}] = {det}, deveria ser 1.0"
        
        # Verificar se R é uma matriz ortogonal (R^T * R = I)
        product = R[i].T @ R[i]
        assert np.allclose(product, np.eye(3), atol=1e-5), f"R[{i}] não é ortogonal"
        
    # Analisar a função tangentPlane
    P = tangentPlane(len(pts), T, B, N)
    
    # Verificar se as matrizes P estão corretas (cada coluna deve ser T, B, N)
    for i in range(len(pts)):
        expected_P = np.vstack([N[i], T[i], B[i]])
        assert np.allclose(P[i], expected_P, atol=1e-5), f"P[{i}] incorreta"
    
    print("Todas as bases são ortonormais e orientadas corretamente!")
    print("Todas as matrizes de rotação preservam orientação!")