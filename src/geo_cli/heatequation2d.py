import numpy as np
import meshio, sys, os
import polyscope as ps
from scipy.spatial import KDTree

from mesh.pyvet import VET
from mesh.vector_operators import normalize_mesh
from rbf.rbf_fd_operators import compute_surface_operators2d

def main():

    # Input mesh
    mesh = meshio.read('meshes/mesh.obj', file_format='obj')
    # pts = mesh.points
    sft = 0
    pts = normalize_mesh(mesh.points) + sft
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0]

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    # T, B, N = mesh.computeOrthonormalBase()
    # del mesh

    print('Construindo os operadores via RBF-FD...')
    Gx2D, Gy2D, Lc = compute_surface_operators2d(np.hstack((pts[:,0].reshape(-1,1), pts[:,2].reshape(-1,1))))

    ## Para a malha 3 phs, 2 grau do polinomio e 20 vizinhos fica ótimo
    lap2D = Gx2D @ Gx2D + Gy2D @ Gy2D                       # Divergente do Gradiente 2D

    # Source
    source_function = np.zeros(nopts)
    center = np.mean(pts, axis=0)
    center = np.argmin(np.linalg.norm(pts - center, axis=1))
    source = [center] +  mesh.compute_k_ring(center,1)

    source_function = np.zeros(nopts)

    # Applying Gaussian
    source_heat = source_function.copy()
    source_heat[source] = 1
    source_heat = np.exp(-np.linalg.norm(pts - pts[center,:], axis=1)**2 / 0.05**2)
    source_heat = source_heat / np.max(source_heat)

    t = 0.01
    phi_lap = source_heat.copy()
    phi_lap = np.linalg.solve(np.eye(nopts)-t*lap2D, phi_lap)
    # phi_Lc = np.linalg.solve(np.eye(nopts)-t*Lc_heat, phi_lap)
    for i in range(1,3):
        phi_lap = np.linalg.solve(np.eye(nopts)-t*lap2D, phi_lap)
        # phi_Lc = np.linalg.solve(np.eye(nopts)-t*Lc_heat, phi_Lc)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Function", source_heat, cmap='turbo')
    ps_mesh.add_scalar_quantity("Heat Equation", phi_lap, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Heat Equation Lc", phi_Lc, cmap='turbo')
    # ps.register_point_cloud("Center", pts[center,:].reshape((1,-1)), radius=0.003, color=(0,0,0))

    # Bounding box mesh
    # mesh1 = meshio.read('meshes/bbox.obj', file_format='obj')
    # pts1 = mesh1.points + sft
    # tri1 = np.array(mesh1.cells_dict['triangle'])
    # ps_mesh1 = ps.register_surface_mesh("Bbox", pts1, tri1, transparency=0.15)

    ps.show()

if __name__ == '__main__': main()