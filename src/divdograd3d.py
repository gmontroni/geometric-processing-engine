import numpy as np
import meshio, sys, os
import polyscope as ps

from geopackages.vet import VET
from geopackages.rbf import compute_surface_operators3d


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

    print('Construindo os operadores via RBF-FD...')
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators3d(pts, T, B, N)

    lap3D = Gx3D @ Gx3D + Gy3D @ Gy3D + Gz3D @ Gz3D       # Divergente do Gradiente 3D

    source = 100

    ## Função contínua suave com foco no centro (lap3d)
    f = np.zeros(nopts)
    ring = mesh.compute_ring(source)
    f[ring] = 1.0
    f[source] = 1.0
    f = np.exp(-np.linalg.norm(pts - pts[source,:], axis=1)**2 / 0.1**2)
    f = f / np.max(f)

    ## Lc como operador de Laplaciano
    # f = np.zeros(nopts)
    # center = np.mean(pts, axis=0)
    # center = np.argmin(np.linalg.norm(pts - center, axis=1))
    # ring = mesh.compute_ring(center)
    # f[center] = 1.0

    print('Equação do calor')
    t = 0.01
    u = f.copy()
    u = np.linalg.solve(np.eye(nopts)-t*lap3D, u)
    # for i in range(1,6):
    #     u = np.linalg.solve(np.eye(nopts)-t*Lc, u)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_vector_quantity("Normal", N, radius = 0.020)
    ps_mesh.add_scalar_quantity("Função", f, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação do Calor", u, cmap='turbo')
    # ps.register_point_cloud("Ponto_fonte", pts[source,:].reshape((1,-1)), radius=0.020, color=(0,0,0))

    ps.show()


if __name__ == '__main__': main()