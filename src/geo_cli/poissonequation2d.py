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
    sft = 0.5
    pts = normalize_mesh(mesh.points) + sft
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0] 

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    # del mesh

    print('Construindo os operadores via RBF-FD...')
    Gx2D, Gy2D, Lc = compute_surface_operators2d(np.hstack((pts[:,0].reshape(-1,1), pts[:,2].reshape(-1,1))))
    lap2D = Gx2D @ Gx2D + Gy2D @ Gy2D       # Divergente do Gradiente 2D

    print('Exemplo: Problema de Poisson com condições de Dirichlet (Exemplo 39.1)')

    ## Função fonte f(x,y) = -5pi²/4 sin(pi*x)cos(pi * y/2)
    # source_function = np.zeros(nopts)
    source_function = -5 * np.pi**2 / 4 * np.sin(np.pi * pts[:,0]) * np.cos(np.pi * pts[:,2] / 2)

    ## Solução exata u(x,y) = sin(pi * x)cos(pi * y/2)
    exact_solution = np.sin(np.pi * pts[:,0]) * np.cos(np.pi * pts[:,2] / 2)
    tolerance = 1e-6
    boundary_indices = []

    ## Condições de fronteira
    # y = 0 (onde u = sin(pi * x))
    gamma1_indices = np.where(np.abs(pts[:,2]) < tolerance)[0]

    # gamma2_indices = np.where(np.abs(pts[:,2] - 1) < tolerance)[0]

    gamma2_indices = np.where(
        (np.abs(pts[:,0]) < tolerance) |      # x = 0
        (np.abs(pts[:,0] - 1) < tolerance) |  # x = 1
        (np.abs(pts[:,2] - 1) < tolerance)    # y = 1
    )[0]

    print('via Laplaciano')
    # Resolver via Laplaciano Lc
    Lc_dirichlet = Lc.copy()
    source_dirichlet = source_function.copy()

    ## Condições de contorno
    for idx in gamma1_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        Lc_dirichlet[idx,:] = row
        source_dirichlet[idx] = np.sin(np.pi * pts[idx,0])

    for idx in gamma2_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        Lc_dirichlet[idx,:] = row
        source_dirichlet[idx] = 0

    phi_dirichlet_lap = np.linalg.solve(Lc_dirichlet, source_dirichlet)

    print('via Divergente do Gradiente')
    ## Resolver via Div do Gradiente
    laplaciano_dirichlet = (Gx2D @ Gx2D + Gy2D @ Gy2D).copy()
    source_divgrad = source_function.copy()

    ## Condições de contorno
    for idx in gamma1_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        laplaciano_dirichlet[idx,:] = row
        source_divgrad[idx] = np.sin(np.pi * pts[idx,0])

    for idx in gamma2_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        laplaciano_dirichlet[idx,:] = row
        source_divgrad[idx] = 0

    phi_dirichlet_divgrad = np.linalg.solve(laplaciano_dirichlet, source_divgrad)

    erroLc = np.linalg.norm(exact_solution - phi_dirichlet_lap)
    print(erroLc)
    erroDivGrad = np.linalg.norm(exact_solution - phi_dirichlet_divgrad)
    print(erroDivGrad)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Problema Dirichlet (Laplaciano)", phi_dirichlet_lap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Problema Dirichlet (DivGrad)", phi_dirichlet_divgrad, cmap='turbo')
    ps_mesh.add_scalar_quantity("Solução Exata Dirichlet", exact_solution, cmap='turbo')
    ps_mesh.add_scalar_quantity("Função Fonte Dirichlet", source_function, cmap='turbo')

    # Bounding box mesh
    mesh1 = meshio.read('meshes/bbox.obj', file_format='obj')
    pts1 = mesh1.points + sft
    tri1 = np.array(mesh1.cells_dict['triangle'])
    ps_mesh1 = ps.register_surface_mesh("Bbox", pts1, tri1, transparency=0.15)

    ps.show()

if __name__ == '__main__': main()