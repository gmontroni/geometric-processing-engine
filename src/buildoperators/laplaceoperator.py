import numpy as np
from mesh import VET
from mesh.vector_operators import normalize_mesh
from rbf.rbf_fd_operators import compute_surface_operators

def build_laplace_operators(mesh_path):
    # Load mesh
    import meshio
    mesh = meshio.read(mesh_path, file_format='obj')
    pts = normalize_mesh(mesh.points)
    tri = np.array(mesh.cells_dict['triangle'])
    nopts = pts.shape[0]

    # Local coordinates
    mesh_vet = VET(pts, tri)
    T, B, N = mesh_vet.computeOrthonormalBase()

    # Build Laplacian
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators(pts, T, B)
    lap3D = Gx3D @ Gx3D + Gy3D @ Gy3D + Gz3D @ Gz3D
    lap3D_heat = lap3D.copy()
    Lc_heat = Lc.copy()

    return lap3D_heat, Lc_heat