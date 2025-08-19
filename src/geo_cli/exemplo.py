import numpy as np
import meshio, sys, os
import polyscope as ps
import scipy.spatial.distance as sd
import numpy.polynomial.polynomial as pp
from scipy.spatial import KDTree

from mesh import VET

def rbf_fd_weights(X, ctr, s, d):
  #   X : each row contains one node in R^2
  # ctr : center (evaluation) node
  # s,d : PHS order and polynomial degree
	
  rbf  = lambda r, s: r**s
  Drbf = lambda r, s, xi: s * xi * r**(s-2)
  Lrbf = lambda r, s: s**2 * r**(s-2)
	
  n = X.shape[0] 
  for i in range(2): X[:,i] -= ctr[i]
  DM = sd.squareform(sd.pdist(X))
  D0 = np.sqrt(X[:,0]**2 + X[:,1]**2)
  A = rbf(DM,s)
  b = np.vstack((Lrbf(D0,s), Drbf(D0,s,-X[:,0]), Drbf(D0,s,-X[:,1]))).T
                 
  if d > -1: #adding polynomials
    m = int((d+2)*(d+1)/2)
    O, k = np.zeros((m,m)), 0
    P, LP = np.zeros((n,m)), np.zeros((m,3))
    PX = pp.polyvander(X[:,0],d)
    PY = pp.polyvander(X[:,1],d)
    for j in range(d+1):
      P[:,k:k+j+1], k = PX[:,j::-1]*PY[:,:j+1], k+j+1
    if d > 0: LP[1,1], LP[2,2] = 1, 1
    if d > 1: LP[3,0], LP[5,0] = 2, 2 
    A = np.block([[A,P],[P.T,O]])
    b = np.block([[b],[LP]])
  	
  # each column contains the weights for 
  # the Laplacian, d/dx1, d/dx2, respectivly.
  weights = np.linalg.solve(A,b)
  return weights[:n,:]

def compute_surface_operators2d(pts):
    # pts: n x 3
    nopts = pts.shape[0]

    # KD-Tree
    tree = KDTree(pts)

    Lc    = np.zeros((nopts, nopts))
    Gx2D  = np.zeros((nopts, nopts))
    Gy2D  = np.zeros((nopts, nopts))

    for i in range(nopts):
        
        _, idx = tree.query(pts[i,:], k=30) # 50 vizinhos mais próximos
        W = rbf_fd_weights(pts[idx,:], pts[i,:], 5, 3) # 5 phs, 5 grau do polinomio
        
        Lc[i, idx] = W[:,0]
        Gx2D[i, idx] = W[:,1]
        Gy2D[i, idx] = W[:,2]
    
    return Gx2D, Gy2D, Lc

def main():

    # Input mesh
    mesh = meshio.read('meshes/mesh.obj', file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0] 

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    del mesh

    print('Construindo os operadores via RBF-FD...')
    Gx2D, Gy2D, Lc = compute_surface_operators2d(np.hstack((pts[:,0].reshape(-1,1), pts[:,2].reshape(-1,1))))

    source = 1784

    print('Exemplo: Equação do calor')

    print('via Divergente do Gradiente')
    ## Equação do calor via Div do Gradiente
    Lap = Gx2D @ Gx2D + Gy2D @ Gy2D
    
    delta = np.zeros(nopts)
    delta[source] = 1
    dt = 0.01
    u = np.linalg.solve(np.eye(nopts) - dt*Lap, delta)
    for i in range(10):
        u = np.linalg.solve(np.eye(nopts) - dt*Lap, u)

    print('via Laplaciano')
    ## Equação do calor via Laplaciano Lc
    deltaLc = np.zeros(nopts)
    deltaLc[source] = 1
    dtLc = 0.01
    uLc = np.linalg.solve(np.eye(nopts) - dtLc*Lc, deltaLc)
    for i in range(10):
        uLc = np.linalg.solve(np.eye(nopts) - dtLc*Lc, uLc)
    

    print('Exemplo: Problema de Poisson com condições de Dirichlet (Exemplo 39.1)')

    ## Função fonte f(x,y) = -5pi²/4 sin(pi*x)cos(pi * y/2)
    source_function = -5 * np.pi**2 / 4 * np.sin(np.pi * pts[:,0]) * np.cos(np.pi * pts[:,2] / 2)

    ## Solução exata u(x,y) = sin(pi * x)cos(pi * y/2)
    exact_solution = np.sin(np.pi * pts[:,0]) * np.cos(np.pi * pts[:,2] / 2)
    tolerance = 1e-6
    boundary_indices = []

    ## Condições de fronteira
    # y = 0 (onde u = sin(pi * x))
    gamma1_indices = np.where(np.abs(pts[:,2]) < tolerance)[0]

    gamma2_indices = np.where(
        (np.abs(pts[:,0]) < tolerance) |  # x = 0
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

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Equação do Calor via Laplaciano", uLc, cmap='turbo') 
    ps_mesh.add_scalar_quantity("Equação do Calor via Divergente do Gradiente", u, cmap='turbo')
    ps_mesh.add_scalar_quantity("Problema Dirichlet (Laplaciano)", phi_dirichlet_lap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Problema Dirichlet (DivGrad)", phi_dirichlet_divgrad, cmap='turbo')
    ps_mesh.add_scalar_quantity("Solução Exata Dirichlet", exact_solution, cmap='turbo')
    ps_mesh.add_scalar_quantity("Função Fonte Dirichlet", source_function, cmap='turbo')
    ps.register_point_cloud("Ponto_fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(0,0,0))
    ps.show()

if __name__ == '__main__': main()