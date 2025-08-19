import numpy as np
import meshio, sys, os
import polyscope as ps
from scipy.spatial import KDTree

from mesh.pyvet import VET
from rbf.rbf_fd_operators import compute_surface_operators3d

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
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0] 

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    # del mesh

    source_function = np.zeros(nopts)
    source = [4102, 4142]                       ## Pontos fontes para o bunny
    # source = [2838, 2718]                       ## Pontos fontes para a esfera

    print('Construindo os operadores via RBF-FD...')
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators3d(pts, T, B, N)
    lap3D = Gx3D @ Gx3D + Gy3D @ Gy3D + Gz3D @ Gz3D       # Divergente do Gradiente 3D

    # Copy para Dirichlet
    lap3D_dirichlet = lap3D.copy()
    Lc_dirichlet = Lc.copy()
    source_dirichlet = source_function.copy()

    # Indices da condição de contorno de Dirichlet
    gamma1_indices = [source[0]] + mesh.compute_k_ring(source[0],2)
    gamma2_indices = [source[1]] + mesh.compute_k_ring(source[1],2)

    ## Condições de contorno de Dirichlet p1 = 1
    for idx in gamma1_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        lap3D_dirichlet[idx,:] = row
        Lc_dirichlet[idx,:] = row
        source_dirichlet[idx] = 2.0

    ## Condições de contorno de Dirichlet p2 = -1
    for idx in gamma2_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        lap3D_dirichlet[idx,:] = row
        Lc_dirichlet[idx,:] = row
        source_dirichlet[idx] = -1.0

    phi_dirichlet_lap = np.linalg.solve(lap3D_dirichlet, source_dirichlet)
    phi_dirichlet_lc = np.linalg.solve(Lc_dirichlet, source_dirichlet)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Função", source_dirichlet, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação de Poisson Div do Grad", phi_dirichlet_lap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação de Poisson", phi_dirichlet_lc, cmap='turbo')
    # ps.register_point_cloud("Vizinhos do ponto fonte", pts[vecIdx[source],:].reshape((-1,3)), radius=0.003, color=(1,0,0))

    ps.show()

if __name__ == '__main__': main()