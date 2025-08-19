import numpy as np
import meshio, sys, os
import polyscope as ps
from scipy.spatial import KDTree

from mesh.pyvet import VET
from mesh.vector_operators import normalize_mesh
from rbf.rbf_fd_operators import compute_surface_operators, compute_surface_operators3d, compute_surface_operators_with_reliability

def main():

    # Input mesh
    # Verificar se um arquivo foi especificado
    if len(sys.argv) <= 1:
        print("Erro: É necessário especificar um arquivo de malha.")
        print("Uso: uv run src/operators3d-tests.py <arquivo.obj>")
        sys.exit(1)
        
    meshName = sys.argv[1]
    if '/' not in meshName:
        fileName = f'meshes/{meshName}'
    else:
        fileName = meshName
    # print(f"Carregando malha: {meshName}")

    mesh = meshio.read(fileName, file_format='obj')
    sft = 2
    # pts = mesh.points
    pts = normalize_mesh(mesh.points) * sft
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0] 

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    # del mesh

    source_function = np.zeros(nopts)
    source = 40                             ## Ponto fonte (centro paraboloide)
    ring = [source] + mesh.compute_k_ring(source,1)

    print('Construindo os operadores via RBF-FD...')
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators(pts, T, B)
    # Gx3D, Gy3D, Gz3D, Lc, _ = compute_surface_operators_with_reliability(pts, T, B, N, k=50, max_neighbors=30, 
    #                                          w_theta=0.34, w_proj=0.33, w_dist=0.33)
    lap3D = Gx3D @ Gx3D + Gy3D @ Gy3D + Gz3D @ Gz3D       # Divergente do Gradiente 3D

    # Copy para Dirichlet
    lap3D_heat = lap3D.copy()
    Lc_heat = Lc.copy()

    # Applying Gaussian
    source_heat = source_function.copy()
    source_heat[ring] = 1
    source_heat = np.exp(-np.linalg.norm(pts - pts[source,:], axis=1)**2 / 0.08**2)
    source_heat = source_heat / np.max(source_heat)

    t = 0.01
    epsil = 1e-4
    M_lap = (1 + epsil) * np.eye(nopts) - t * lap3D_heat
    # M_lc = (1 + epsil) * np.eye(nopts) - t * Lc_heat

    phi_lap = source_heat.copy()
    phi_Lc = source_heat.copy()
    phi_lap = np.linalg.solve(M_lap, phi_lap)
    # phi_Lc = np.linalg.solve(M_lc, phi_Lc)
    # phi_lap = np.linalg.solve(np.eye(nopts)-t*lap3D_heat, phi_lap)
    phi_Lc = np.linalg.solve(np.eye(nopts)-t*Lc_heat, phi_Lc)
    for i in range(1,3):
        # phi_lap = np.linalg.solve(np.eye(nopts)-t*lap3D_heat, phi_lap)
        phi_Lc = np.linalg.solve(np.eye(nopts)-t*Lc_heat, phi_Lc)
        phi_lap = np.linalg.solve(M_lap, phi_lap)
        # phi_Lc = np.linalg.solve(M_lc, phi_Lc)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    # ps_mesh1 = ps.register_surface_mesh("Original Mesh", pts1, tri, smooth_shade=True)
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Function", source_heat, cmap='turbo')
    ps_mesh.add_scalar_quantity("Heat Equation", phi_lap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Heat Equation Lc", phi_Lc, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Equação de Poisson", phi_dirichlet_lc, cmap='turbo')
    # ps.register_point_cloud("Vizinhos do ponto fonte", pts[vecIdx[source],:].reshape((-1,3)), radius=0.003, color=(1,0,0))

    # Bounding box mesh
    mesh1 = meshio.read('meshes/bbox.obj', file_format='obj')
    pts1 = mesh1.points * sft
    tri1 = np.array(mesh1.cells_dict['triangle'])
    ps_mesh1 = ps.register_surface_mesh("Bbox", pts1, tri1, transparency=0.15)

    ps.show()

if __name__ == '__main__': main()