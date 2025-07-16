import numpy as np
import meshio, sys, os
import polyscope as ps

from geopackages.vet.pyvet import VET
from geopackages.rbf.rbf_fd_operators import compute_surface_operators2d

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
    
    # Função bem definida
    u = np.sin(pts[:,0]) + np.cos(pts[:,2])

    print('Exemplo 01: Gradiente')
    gradX, gradY = Gx2D @ u, Gy2D @ u
    grad = np.hstack((gradX.reshape(-1,1), gradY.reshape(-1,1)))
    # X2D = grad / np.linalg.norm(grad, axis=1).reshape(-1,1)       # normalizar para visualização, mas não para as contas
    gradient = np.hstack((grad[:,0].reshape(-1,1), np.zeros((nopts,1)), grad[:,1].reshape(-1,1)))  # plot grad 3D
    exactGradient = np.hstack((np.cos(pts[:,0]).reshape(-1,1), np.zeros((nopts,1)), -np.sin(pts[:,2]).reshape(-1,1)))
    # exactGradient = exactGradient / np.linalg.norm(exactGradient, axis=1).reshape(-1,1)      # normalizar para visualização, mas não para as contas

    print('Exemplo 02: Divergente')
    div = Gx2D @ grad[:,0] + Gy2D @ grad[:,1]               # Divergente do Gradiente (Laplaciano)
    exactDiv = - np.sin(pts[:,0]) - np.cos(pts[:,2])
    
    print('Exemplo 03: Laplaciano')
    # u = pts[:,0] + pts[:,2]       # u linear
    laplacian = Lc @ u       

    print('Exemplo 04: Bilaplaciano')
    bilaplacian = Lc @ laplacian
    exactBilaplacian = np.sin(pts[:,0]) + np.cos(pts[:,2])

    print('Exemplo 05: Hessiana')
    nullVec = np.zeros((nopts))
    hessian = np.array([[Gx2D @ gradX, nullVec, Gx2D @ gradY],
                        [nullVec, nullVec, nullVec],
                        [Gy2D @ gradX, nullVec, Gy2D @ gradY]])
    
    exactHessian = np.array([[-np.sin(pts[:,0]), nullVec, nullVec],
                             [nullVec, nullVec, nullVec], 
                             [nullVec, nullVec, -np.cos(pts[:,2])]])
    
    laplacianWithHessian = hessian[0, 0, :] + hessian[1, 1, :] + hessian[2, 2, :]

    print('Exemplo 05: Gradiente do Divergente')
    gradXdiv, gradYdiv = Gx2D @ div, Gy2D @ div
    gradDiv = np.hstack((gradXdiv.reshape(-1,1), gradYdiv.reshape(-1,1)))
    # X2Dgraddiv = gradDiv / np.linalg.norm(gradDiv, axis=1).reshape(-1,1)        # normalizar para visualização, mas não para as contas
    gradDiv = np.hstack((gradDiv[:,0].reshape(-1,1), np.zeros((nopts,1)), gradDiv[:,1].reshape(-1,1))) # plot 3D  

    print('Exemplo 06: Equação do calor')
    source = 1784
    delta = np.zeros(nopts)
    delta[source] = 1
    # delta[[0,10,110,120]] = 1

    t = 0.01
    f = np.linalg.solve(np.eye(nopts) - t*Lc, delta)
    for i in range(1,10):
        f = np.linalg.solve(np.eye(nopts)-t*Lc, f)

    print('Exemplo 07: Equação de Poisson')
    row = np.zeros(nopts)
    row[source] = 1
    Lc[source,:] = row
    div[source] = 0
    phi = np.linalg.solve(Lc, div)

    print('\nCalculando os erros...\n')

    # Erro do Gradiente (Visual)
    normalizedGrad = gradient / np.linalg.norm(gradient, axis=1).reshape(-1,1)
    normalizedExactGrad = exactGradient / np.linalg.norm(exactGradient, axis=1).reshape(-1,1) 
    graderror = np.abs(1 - np.sum   (normalizedGrad*normalizedExactGrad, axis=1))       

    print('Erros calculados na norma infinito...')

    erroGrad = np.abs(exactGradient - gradient)
    infNormGrad = np.max(erroGrad)
    print('O erro do Gradiente na norma infinita é: ', infNormGrad)

    erroDiv = np.abs(exactDiv - div)
    infNormDiv = np.max(erroDiv)
    print('O erro do Divergente na norma infinita é: ', infNormDiv)

    erroLap = np.abs(exactDiv - laplacian)
    infNOrmLap = np.max(erroLap)
    print('O erro do Laplaciano na norma infinita é: ', infNOrmLap)

    erroBilap = np.abs(exactBilaplacian - bilaplacian)
    infNOrmBilap = np.max(erroBilap)
    print('O erro do Bilaplaciano na norma infinita é: ', infNOrmBilap)

    erroHess = np.abs(exactHessian - hessian)
    infNormHess = np.max(erroHess)
    print('O erro do Hessiana na norma infinita é: ', infNormHess)

    print('\nErros calculados na norma do erro quadrático médio (MSE)...')

    mseGrad = np.mean(erroGrad**2)
    print('O erro quadrático médio (MSE) do Gradiente é: ', mseGrad)

    mseDiv = np.mean(erroDiv**2)
    print('O erro quadrático médio (MSE) do Divergente é: ', mseDiv)

    mseLap = np.mean(erroLap**2)
    print('O erro quadrático médio (MSE) do Laplaciano é: ', mseLap)

    mseBilap = np.mean(erroBilap**2)
    print('O erro quadrático médio (MSE) do Bilaplaciano é: ', mseBilap)

    mseHess = np.mean(erroHess**2)
    print('O erro quadrático médio (MSE) da Hessiana é: ', mseHess)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Função", u, cmap='turbo')
    ps_mesh.add_vector_quantity("Gradiente", gradient)
    ps_mesh.add_vector_quantity("Gradiente Analítico", exactGradient)
    ps_mesh.add_vector_quantity("Gradiente do Divergente", gradDiv)
    ps_mesh.add_scalar_quantity("Divergente", div, cmap='turbo')
    ps_mesh.add_scalar_quantity("Divergente Analítico", exactDiv, cmap='turbo')
    ps_mesh.add_scalar_quantity("Laplaciano", laplacian, cmap='turbo') 
    ps_mesh.add_scalar_quantity("Laplaciano gerado pela Hessiana", laplacianWithHessian, cmap='turbo') 
    ps_mesh.add_scalar_quantity("Bilaplaciano", bilaplacian, cmap='turbo')         
    ps_mesh.add_scalar_quantity("Bilaplaciano Analítico", exactBilaplacian, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação do Calor", f, cmap='turbo')
    ps_mesh.add_scalar_quantity("Erro do Gradiente", graderror, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equacao de Poisson", phi, cmap='turbo')
    ps.register_point_cloud("Ponto_fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(0,0,0))

    ps.show()

if __name__ == '__main__': main()