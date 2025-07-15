import numpy as np
import meshio, sys, os
import polyscope as ps

sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'src'))
from pyvet import VET, normalize
from rbf_fd_operators import compute_surface_operators

def main():

    # Input mesh
    mesh = meshio.read('input/bunny.obj', file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0] 

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    del mesh

    print('Construindo os operadores via RBF-FD...')
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators(pts, T, B)

    print('Exemplo 01: Equação do calor')
    source = 3630
    delta = np.zeros(nopts)
    delta[source] = 1
    #t = 1
    #u = np.linalg.solve(np.eye(nopts) - t * Lc, delta)

    t = 0.01
    u = np.linalg.solve(np.eye(nopts) - t*Lc, delta)
    for i in range(1,6):
        # t = i * t
        u = np.linalg.solve(np.eye(nopts)-t*Lc, u)
    
    # u bem definida

    print('Exemplo 02: Gradiente')
    gradX = Gx3D @ u
    gradY = Gy3D @ u
    gradZ = Gz3D @ u
    grad = np.hstack((gradX.reshape(-1,1), gradY.reshape(-1,1), gradZ.reshape(-1,1)))
    X3D = -normalize(grad)

    print('Exemplo 03: Divergente')
    div = Gx3D @ X3D[:,0] + Gy3D @ X3D[:,1] + Gz3D @ X3D[:,2]

    print('Exemplo 04: Equação de Poisson')
    #Lc = 1e-5*np.eye(nopts) + Lc
    row = np.zeros(nopts)
    row[source] = 1
    Lc[source,:] = row
    div[source] = 0
    phi = np.linalg.solve(Lc, div)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Superficie", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Equacao do calor", u, cmap='turbo')
    ps_mesh.add_vector_quantity("Gradiente", X3D)
    ps_mesh.add_scalar_quantity("Divergente", div, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equacao de Poisson", phi, cmap='turbo')
    ps.register_point_cloud("Ponto_fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(0,0,0))

    ps.show()

if __name__ == '__main__': main()