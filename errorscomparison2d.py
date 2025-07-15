import numpy as np
import meshio, sys, os
import polyscope as ps

sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'src'))
from pyvet import VET
from rbf_fd_operators import compute_surface_operators2d

def main():

    # Input mesh
    mesh = meshio.read('input/mesh.obj', file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0] 

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    del mesh

    print('Construindo os operadores via RBF-FD...')
    Gx2D, Gy2D, Lc = compute_surface_operators2d(np.hstack((pts[:,0].reshape(-1,1), pts[:,2].reshape(-1,1))))
    
    ## Função bem definida
    u = np.sin(pts[:,0]) + np.cos(pts[:,2])

    ## Divergente do Gradiente
    divGrad = (Gx2D @ Gx2D + Gy2D @ Gy2D).copy()
    biLDivGrad = (Gx2D @ Gx2D @ Gx2D @ Gx2D + Gy2D @ Gy2D @ Gy2D @ Gy2D).copy()

    ## Exemplo de operadores
    # # Gradiente
    # print('Exemplo 01: Gradiente')
    # gradX, gradY = Gx2D @ u, Gy2D @ u
    # grad = np.hstack((gradX.reshape(-1,1), gradY.reshape(-1,1)))
    # # X2D = grad / np.linalg.norm(grad, axis=1).reshape(-1,1)       # normalizar para visualização, mas não para as contas
    # gradient = np.hstack((grad[:,0].reshape(-1,1), np.zeros((nopts,1)), grad[:,1].reshape(-1,1)))  # plot grad 3D
    # exactGradient = np.hstack((np.cos(pts[:,0]).reshape(-1,1), np.zeros((nopts,1)), -np.sin(pts[:,2]).reshape(-1,1)))
    # # exactGradient = exactGradient / np.linalg.norm(exactGradient, axis=1).reshape(-1,1)      # normalizar para visualização, mas não para as contas

    # Laplaciano
    print('Exemplo 02: Laplaciano')
    # u = pts[:,0] + pts[:,2]       # u linear
    lap = Lc @ u
    exactLap = - np.sin(pts[:,0]) - np.cos(pts[:,2])

    # Laplaciano usando o divergente do gradiente
    print('Exemplo 03: Laplaciano via DivGrad')
    # div = Gx2D @ grad[:,0] + Gy2D @ grad[:,1]               # Divergente do Gradiente (Laplaciano)
    div = divGrad @ u                                         # Divergente do Gradiente (Laplaciano)
    exactDiv = - np.sin(pts[:,0]) - np.cos(pts[:,2])

    # Bilaplaciano Lc @ (Lc @ u)
    print('Exemplo 04: Bilaplaciano')
    # biLaplacian = Lc @ lap
    biLaplacian = Lc @ Lc @ u
    exactBilaplacian = np.sin(pts[:,0]) + np.cos(pts[:,2])

    # Bilaplaciano usando divergente do gradiente duas vezes div(grad(div(grad(u)))) 
    print('Exemplo 05: Bilaplaciano via DivGrad duplo')
    
    # gradLap_x = Gx2D @ div  # gradiente x do laplaciano
    # gradLap_y = Gy2D @ div  # gradiente y do laplaciano
    # biLaplacianDivGrad = Gx2D @ gradLap_x + Gy2D @ gradLap_y
    biLaplacianDivGrad = biLDivGrad @ u

    print('Exemplo 06: Hessiana')
    # H = [∂²u/∂x²   ∂²u/∂x∂y]
    #     [∂²u/∂y∂x  ∂²u/∂y² ]

    hessian_xx, hessian_yy, hessian_xy = (Gx2D @ Gx2D) @ u, (Gy2D @ Gy2D) @ u, (Gx2D @ Gy2D) @ u  
    
    hessian = np.array([[hessian_xx, hessian_xy],
                        [hessian_xy, hessian_yy]])

    # Para visualização/cálculos:
    laplacianFromHessian = hessian_xx + hessian_yy  # Traço da Hessiana = Laplaciano

    exactHessian_xx, exactHessian_yy, exactHessian_xy  = -np.sin(pts[:,0]), -np.cos(pts[:,2]), np.zeros(nopts)
    exactHessian = np.array([[exactHessian_xx, exactHessian_xy],
                            [exactHessian_xy, exactHessian_yy]])
    
    # print('\n\n=== ANÁLISE DOS OPERADORES ===')

    # # Diferença entre as matrizes
    # matriz_diff = np.max(np.abs(Lc - divGrad))
    # print('Diferença máxima entre matrizes: ', matriz_diff)

    # # Diferença quando aplicadas à função u
    # resultado_diff = np.max(np.abs(Lc @ u - divGrad @ u))
    # print('Diferença máxima nos resultados: ', resultado_diff)

    # # Análise de condicionamento
    # cond_Lc = np.linalg.cond(Lc)
    # cond_div = np.linalg.cond(divGrad)
    # print('Número de condição Lc: ', cond_Lc)
    # print('Número de condição DivGrad: ', cond_div)

    # # Análise espectral
    # eigvals_Lc = np.linalg.eigvals(Lc)
    # eigvals_div = np.linalg.eigvals(divGrad)
    # print('Autovalores extremos Lc: ', np.min(np.real(eigvals_Lc)), np.max(np.real(eigvals_Lc)))
    # print('Autovalores extremos DivGrad: ', np.min(np.real(eigvals_div)), np.max(np.real(eigvals_div)))

    # # Análise de esparsidade
    # sparsity_Lc = np.sum(np.abs(Lc) > 1e-12) / (nopts * nopts)
    # sparsity_div = np.sum(np.abs(divGrad) > 1e-12) / (nopts * nopts)
    # print('Densidade Lc: ', sparsity_Lc)
    # print('Densidade DivGrad: ', sparsity_div)

    # print('\n\n=== DIAGNÓSTICO DIV DO GRAD ===')

    # # Verificar se há NaN ou inf na matriz
    # has_nan = np.any(np.isnan(divGrad))
    # has_inf = np.any(np.isinf(divGrad))
    # print(f"Contém NaN: {has_nan}")
    # print(f"Contém inf: {has_inf}")

    # # Valores extremos
    # print(f"Valor máximo: {np.max(divGrad)}")
    # print(f"Valor mínimo: {np.min(divGrad)}")

    # # Diagnóstico do operador
    # print('Determinante do operador DivGrad: ', np.linalg.det(divGrad))
    # print('Rank do operador DivGrad: ', np.linalg.matrix_rank(divGrad))
    # print('Rank completo seria: ', nopts)

    # # Verificar se é singular
    # singular_values = np.linalg.svd(divGrad, compute_uv=False)
    # print('Menores valores singulares: ', np.sort(singular_values)[:10])
    # print('Número de condição: ', np.linalg.cond(divGrad))

    # # Verificar se há linhas/colunas nulas
    # zero_rows = np.where(np.all(np.abs(divGrad) < 1e-12, axis=1))[0]
    # zero_cols = np.where(np.all(np.abs(divGrad) < 1e-12, axis=0))[0]
    # print(f'Linhas quase nulas: {len(zero_rows)} linhas')
    # print(f'Colunas quase nulas: {len(zero_cols)} colunas')

    # if len(zero_rows) > 0:
    #     print(f'Índices das linhas nulas: {zero_rows[:10]}')  # mostra primeiras 10

    # print('\n\n=== DIAGNÓSTICO LAPLACIANO ===')

    # # Verificar se há NaN ou inf na matriz
    # has_nanLc = np.any(np.isnan(Lc))
    # has_infLc = np.any(np.isinf(Lc))
    # print(f"Contém NaN: {has_nanLc}")
    # print(f"Contém inf: {has_infLc}")

    # # Valores extremos
    # print(f"Valor máximo: {np.max(Lc)}")
    # print(f"Valor mínimo: {np.min(Lc)}")

    # # Diagnóstico do operador
    # print('Determinante do operador Lc: ', np.linalg.det(Lc))
    # print('Rank do operador Lc: ', np.linalg.matrix_rank(Lc))
    # print('Rank completo seria: ', nopts)

    # # Verificar se é singular
    # singular_valuesLc = np.linalg.svd(Lc, compute_uv=False)
    # print('Menores valores singulares: ', np.sort(singular_valuesLc)[:10])
    # print('Número de condição: ', np.linalg.cond(Lc))

    # # Verificar se há linhas/colunas nulas
    # zero_rowsLc = np.where(np.all(np.abs(Lc) < 1e-12, axis=1))[0]
    # zero_colsLc = np.where(np.all(np.abs(Lc) < 1e-12, axis=0))[0]
    # print(f'Linhas quase nulas: {len(zero_rowsLc)} linhas')
    # print(f'Colunas quase nulas: {len(zero_rowsLc)} colunas')

    # if len(zero_rowsLc) > 0:
    #     print(f'Índices das linhas nulas: {zero_rowsLc[:10]}')  # mostra primeiras 10

    source = 1784
    delta = np.zeros(nopts)
    delta[source] = 1
    print('Exemplo 07: Equação do calor via Laplaciano')
    t = 0.01
    f = np.linalg.solve(np.eye(nopts) - t*Lc, delta)
    for i in range(1,5):
        f = np.linalg.solve(np.eye(nopts)-t*Lc, f)

    ### FAZER .COPY EM TUDO AMANHA
    print('Exemplo 08: Equação do calor via Divergente do Gradiente')
    divdoGrad = (Gx2D @ Gx2D + Gy2D @ Gy2D).copy()

    delta_divgrad = np.zeros(nopts)
    delta_divgrad[source] = 1
    t = 0.01
    f_divgrad = np.linalg.solve(np.eye(nopts)-t*divdoGrad, delta_divgrad)
    for i in range(1,5):
        f_divgrad = np.linalg.solve(np.eye(nopts)-t*divdoGrad, f_divgrad)

    print('Exemplo 09: Equação de Poisson via Laplaciano')
    row = np.zeros(nopts)
    row[source] = 1
    Lc[source,:] = row
    div[source] = 0
    phi0 = np.linalg.solve(Lc, div)

    print('Exemplo 10: Equação de Poisson via Divergente do Gradiente')
    row = np.zeros(nopts)
    row[source] = 1
    Lc[source,:] = row
    div[source] = 0
    laplaciano = Gx2D @ Gx2D + Gy2D @ Gy2D
    phi1 = np.linalg.solve(laplaciano, div)

    print('Exemplo 11: Problema de Poisson com condições de Dirichlet (Exemplo 39.1)')

    # Definir a função fonte f(x,y) = -5π²/4 sin(πx)cos(πy/2)
    source_function = -5 * np.pi**2 / 4 * np.sin(np.pi * pts[:,0]) * np.cos(np.pi * pts[:,2] / 2)

    # Solução exata u(x,y) = sin(πx)cos(πy/2)
    exact_solution = np.sin(np.pi * pts[:,0]) * np.cos(np.pi * pts[:,2] / 2)

    # Identificar pontos da fronteira
    # Assumindo que a malha está no domínio [0,1] x [0,1]
    tolerance = 1e-6
    boundary_indices = []

    # Fronteira Γ₁: y = 0 (onde u = sin(πx))
    gamma1_indices = np.where(np.abs(pts[:,2]) < tolerance)[0]
    # Fronteira Γ₂: resto da fronteira (onde u = 0)
    gamma2_indices = np.where(
        (np.abs(pts[:,0]) < tolerance) |  # x = 0
        (np.abs(pts[:,0] - 1) < tolerance) |  # x = 1
        (np.abs(pts[:,2] - 1) < tolerance)    # y = 1
    )[0]

    # Resolver via Laplaciano
    Lc_dirichlet = Lc.copy()
    source_dirichlet = source_function.copy()

    # Aplicar condições de contorno Γ₁: u = sin(πx)
    for idx in gamma1_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        Lc_dirichlet[idx,:] = row
        source_dirichlet[idx] = np.sin(np.pi * pts[idx,0])

    # Aplicar condições de contorno Γ₂: u = 0
    for idx in gamma2_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        Lc_dirichlet[idx,:] = row
        source_dirichlet[idx] = 0

    phi_dirichlet_lap = np.linalg.solve(Lc_dirichlet, source_dirichlet)

    # Resolver via Divergente do Gradiente
    laplaciano_dirichlet = (Gx2D @ Gx2D + Gy2D @ Gy2D).copy()
    source_divgrad = source_function.copy()

    # Aplicar condições de contorno Γ₁
    for idx in gamma1_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        laplaciano_dirichlet[idx,:] = row
        source_divgrad[idx] = np.sin(np.pi * pts[idx,0])

    # Aplicar condições de contorno Γ₂
    for idx in gamma2_indices:
        row = np.zeros(nopts)
        row[idx] = 1
        laplaciano_dirichlet[idx,:] = row
        source_divgrad[idx] = 0

    phi_dirichlet_divgrad = np.linalg.solve(laplaciano_dirichlet, source_divgrad)

    ## Erros
    print('\nCalculando os erros...\n')

    # # Erro do Gradiente (Visual)
    # normalizedGrad = gradient / np.linalg.norm(gradient, axis=1).reshape(-1,1)
    # normalizedExactGrad = exactGradient / np.linalg.norm(exactGradient, axis=1).reshape(-1,1) 
    # graderror = np.abs(1 - np.sum   (normalizedGrad*normalizedExactGrad, axis=1))       

    # Erros norma infinito
    print('Erros calculados na norma infinito...')

    # erroGrad = np.abs(exactGradient - gradient)
    # infNormGrad = np.max(erroGrad)
    # # print('O erro do Gradiente na norma infinita é: ', infNormGrad)

    erroLap = np.abs(exactLap - lap)
    infNOrmLap = np.max(erroLap)
    print('O erro do Laplaciano           na norma infinita é: ', infNOrmLap)

    erroDiv = np.abs(exactDiv - div)
    infNormDiv = np.max(erroDiv)
    print('O erro do Laplaciano (DivGrad) na norma infinita é: ', infNormDiv)

    erroBilap = np.abs(exactBilaplacian - biLaplacian)
    infNormBilap = np.max(erroBilap)
    print('O erro do Bilaplaciano           na norma infinita é: ', infNormBilap)

    erroBilapDivGrad = np.abs(exactBilaplacian - biLaplacianDivGrad)
    infNormBilapDivGrad = np.max(erroBilapDivGrad)
    print('O erro do Bilaplaciano (DivGrad) na norma infinita é: ', infNormBilapDivGrad)

    erroHessian = np.abs(exactHessian - hessian)
    infNormHessian = np.max(erroHessian)
    print('O erro da Hessiana (DivGrad) na norma infinita é: ', infNormHessian)

    # Calcular erros para o problema de Dirichlet
    erroDirichletLap = np.abs(exact_solution - phi_dirichlet_lap)
    infNormDirichletLap = np.max(erroDirichletLap)
    print('O erro do Problema de Dirichlet (Laplaciano) na norma infinita é: ', infNormDirichletLap)

    erroDirichletDivGrad = np.abs(exact_solution - phi_dirichlet_divgrad)
    infNormDirichletDivGrad = np.max(erroDirichletDivGrad)
    print('O erro do Problema de Dirichlet (DivGrad) na norma infinita é: ', infNormDirichletDivGrad)

    # Erros MSE
    print('\nErros calculados na norma do erro quadrático médio (MSE)...')

    # mseGrad = np.mean(erroGrad**2)
    # print('O erro quadrático médio (MSE) do Gradiente é: ', mseGrad)

    mseLap = np.mean(erroLap**2)
    print('O erro quadrático médio (MSE) do Laplaciano           é: ', mseLap)

    mseDiv = np.mean(erroDiv**2)
    print('O erro quadrático médio (MSE) do Laplaciano (DivGrad) é: ', mseDiv)

    mseBilap = np.mean(erroBilap**2)
    print('O erro quadrático médio (MSE) do Bilaplaciano           é: ', mseBilap)

    mseBilapDivGrad = np.mean(erroBilapDivGrad**2)
    print('O erro quadrático médio (MSE) do Bilaplaciano (DivGrad) é: ', mseBilapDivGrad)

    mseHessian = np.mean(erroHessian**2)
    print('O erro quadrático médio (MSE) da Hessiana (DivGrad) é: ', mseHessian)

    mseDirichletLap = np.mean(erroDirichletLap**2)
    print('O erro quadrático médio (MSE) do Problema de Dirichlet (Laplaciano) é: ', mseDirichletLap)

    mseDirichletDivGrad = np.mean(erroDirichletDivGrad**2)
    print('O erro quadrático médio (MSE) do Problema de Dirichlet (DivGrad) é: ', mseDirichletDivGrad)

    consistency_check = np.max(np.abs((Lc @ u) - (hessian_xx + hessian_yy)))
    print('\n\nConsistência Laplaciano vs Traço Hessiana: ', consistency_check)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Função", u, cmap='turbo')
    # ps_mesh.add_vector_quantity("Gradiente", gradient)
    # ps_mesh.add_vector_quantity("Gradiente Analítico", exactGradient)
    # ps_mesh.add_vector_quantity("Gradiente do Divergente", gradDiv)
    ps_mesh.add_scalar_quantity("Divergente do Gradiente", div, cmap='turbo')
    ps_mesh.add_scalar_quantity("Divergente do Gradiente Analítico", exactDiv, cmap='turbo')
    ps_mesh.add_scalar_quantity("Laplaciano", lap, cmap='turbo') 
    ps_mesh.add_scalar_quantity("Laplaciano Analítico", exactLap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Laplaciano gerado pela Hessiana", laplacianFromHessian, cmap='turbo') 
    ps_mesh.add_scalar_quantity("Bilaplaciano", biLaplacian, cmap='turbo')         
    ps_mesh.add_scalar_quantity("Bilaplaciano Analítico", exactBilaplacian, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Equacao de Poisson via Laplaciano", phi0, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Equacao de Poisson via Divergente do Gradiente", phi1, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação do Calor via Laplaciano", f, cmap='turbo') 
    ps_mesh.add_scalar_quantity("Equação do Calor via Divergente do Gradiente", f_divgrad, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Erro do Gradiente", graderror, cmap='turbo') 


    ps_mesh.add_scalar_quantity("Problema Dirichlet (Laplaciano)", phi_dirichlet_lap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Problema Dirichlet (DivGrad)", phi_dirichlet_divgrad, cmap='turbo')
    ps_mesh.add_scalar_quantity("Solução Exata Dirichlet", exact_solution, cmap='turbo')
    ps_mesh.add_scalar_quantity("Função Fonte Dirichlet", source_function, cmap='turbo')
    ps_mesh.add_scalar_quantity("Erro Dirichlet (Laplaciano)", erroDirichletLap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Erro Dirichlet (DivGrad)", erroDirichletDivGrad, cmap='turbo')


    # ps_mesh.add_scalar_quantity("Div do Grad de u", divdoGrad_u, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Equacao de Poisson", phi, cmap='turbo')
    ps.register_point_cloud("Ponto_fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(0,0,0))

    ps.show()

if __name__ == '__main__': main()