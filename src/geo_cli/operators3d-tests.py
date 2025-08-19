import numpy as np
import meshio, sys, os
import polyscope as ps
from scipy.spatial import KDTree

from mesh.pyvet import VET, normalize
from mesh.vector_operators import surfProjection
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
    del mesh

    print('Construindo os operadores via RBF-FD...')
    # Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators(pts, T, B)
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators3d(pts, T, B, N)
    # Gx3D, Gy3D, Gz3D, Lc, vecIdx = compute_surface_operators_with_reliability(pts, T, B, N)

    # Função bem definida
    # u = pts[:,0]
    # u = pts[:,0]**2
    u = np.exp(pts[:,0])
    
    # Gradiente exato
    # exactGradient = np.hstack((np.ones((nopts)).reshape(-1, 1), 
    #                            np.zeros((nopts)).reshape(-1,1), 
    #                            np.zeros((nopts)).reshape(-1,1)))
    
    # exactGradient = np.hstack((2*pts[:, 0].reshape(-1, 1), 
    #                            np.zeros((nopts)).reshape(-1, 1), 
    #                            np.zeros((nopts)).reshape(-1, 1)))

    
    exactGradient = np.hstack((np.exp(pts[:, 0]).reshape(-1, 1),
                                np.zeros(nopts).reshape(-1, 1),
                                np.zeros(nopts).reshape(-1, 1)))
    
    exactGradient = surfProjection(nopts,exactGradient, N)

    # Laplaciano exato
    # exactLapla = -2 * pts[:,0]
    # exactLapla = 2 - 6*u
    exactLapla = (1 - 2*pts[:,0] - pts[:,0]**2) * u

    print('Exemplo 01: Gradiente')
    gradX, gradY, gradZ = Gx3D @ u, Gy3D@ u, Gz3D @ u
    grad = np.hstack((gradX.reshape(-1,1), gradY.reshape(-1,1), gradZ.reshape(-1,1)))
    gradient = -normalize(grad)

    print('Exemplo 02: Laplaciano')
    div = Gx3D @ grad[:,0] + Gy3D @ grad[:,1] + Gz3D @ grad[:,2]                    
    Lapla = Lc @ u
    
    print('Exemplo 03: Equação do calor')
    source = 0
    delta = np.zeros(nopts)
    delta[source] = 1

    t = 0.01
    f = np.linalg.solve(np.eye(nopts) - t*Lc, delta)
    f0 = f.copy()
    for i in range(1,6):
        f = np.linalg.solve(np.eye(nopts)-t*Lc, f)

    # Cálculo dos Erros
    print('\nCalculando os erros...\n')

    # Erro do Gradiente (Visual)
    normalizedGrad = gradient / np.linalg.norm(gradient, axis=1).reshape(-1,1)
    normalizedExactGrad = exactGradient / np.linalg.norm(exactGradient, axis=1).reshape(-1,1) 
    graderror = np.abs(1 - np.sum   (normalizedGrad*normalizedExactGrad, axis=1))       

    print('Erros calculados na norma infinito...')

    erroGrad = np.abs(exactGradient - gradient)
    infNormGrad = np.max(erroGrad)
    print('O erro do Gradiente na norma infinita é: ', infNormGrad)

    erroDiv = np.abs(exactLapla - div)
    infNOrmDiv = np.max(erroDiv)
    print('O erro do Divergente na norma infinita é: ', infNOrmDiv)

    erroLap = np.abs(exactLapla - Lapla)
    infNOrmLap = np.max(erroLap)
    print('O erro do Laplaciano na norma infinita é: ', infNOrmLap)

    print('\nErros calculados na norma do erro quadrático médio (MSE)...')

    mseGrad = np.mean(erroGrad**2)
    print('O erro quadrático médio (MSE) do Gradiente é: ', mseGrad)

    mseDiv = np.mean(erroDiv**2)
    print('O erro quadrático médio (MSE) do Divergente é: ', mseDiv)

    mseLap = np.mean(erroLap**2)
    print('O erro quadrático médio (MSE) do Laplaciano é: ', mseLap)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Função", u, cmap='turbo')
    ps_mesh.add_scalar_quantity("f0 eq Calor", f0, cmap='turbo')
    ps_mesh.add_scalar_quantity("Solução eq Calor", f, cmap='turbo')
    ps_mesh.add_vector_quantity("Gradiente Exato", exactGradient)
    ps_mesh.add_vector_quantity("Gradiente", gradient)
    ps_mesh.add_scalar_quantity("Divergente do Gradiente", div, cmap='turbo')
    ps_mesh.add_scalar_quantity("Laplaciano Exato", exactLapla, cmap='turbo')
    ps_mesh.add_scalar_quantity("Laplaciano", Lapla, cmap='turbo')
    ps_mesh.add_scalar_quantity("Erro do Gradiente", graderror, cmap='turbo')
    ps.register_point_cloud("Ponto fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(0,0,0))
    # ps.register_point_cloud("Vizinhos do ponto fonte", pts[vecIdx[source],:].reshape((-1,3)), radius=0.003, color=(1,0,0))

    ps.show()

if __name__ == '__main__': main()