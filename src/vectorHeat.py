import numpy as np
import copy
import meshio, sys, os
import polyscope as ps
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye, csr_matrix

sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'src'))
from pyvet import VET
from parallelTransport import parallelTransportv2
from vector_heat_functions import buildSourceVectors, computeConnectionLaplacian, buildY0, buildVecHeatOperator, solveVectorHeat
from rbf_fd_operators import compute_surface_operators3d, compute_surface_operators3dd, compute_surface_operators_with_reliability

def main():

    # Input mesh
    # Verificar se um arquivo foi especificado
    if len(sys.argv) <= 1:
        print("Erro: É necessário especificar um arquivo de malha.")
        print("Uso: python3 vectorHeat.py <arquivo.obj>")
        sys.exit(1)
        
    meshName = sys.argv[1]
    # meshName = 'torus.obj'
    if '/' not in meshName:
        fileName = f'input/{meshName}'
    else:
        fileName = meshName
    # print(f"Carregando malha: {meshName}")

    mesh = meshio.read(fileName, file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0]

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()             # T e B criados usando normal
    tangentBasis = np.array([T,B])
    del mesh

    # Transporte Paralelo
    # V = parallelTransportv2(pts,T,B,N,source_point)         # usar sem a conectividade (não tem diferença)

    # Operadores
    Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators3d(pts, T, B, N) 
    # Gx3D, Gy3D, Gz3D, Lc, vecIdx = compute_surface_operators_with_reliability(pts, T, B, N)

    # Sources
    # sourceIdxs = [600, 1560, 2520, 3528, 3000, 2039, 72, 1032]

    # sourceVs = [
    # np.array([-0.0683853626, 0.997658968]),
    # np.array([-0.0989825577, 0.995089114]),
    # np.array([-0.0809355006, -0.996719301]),
    # np.array([-0.0456265137, -0.998958528]),
    # np.array([-0.0295587238, -0.999562979]),
    # np.array([-0.999998152, -0.0018937682]),
    # np.array([-0.999999583, -0.000849745411]),
    # np.array([-0.108760782, 0.994067907])
    # ]

    sourceIdxs = [600]
    sourceVs = [np.array([-0.0456265137, -0.998958528])]
    sources, sourceVectors = buildSourceVectors(nopts, sourceIdxs, sourceVs)  # [(idx, vec), ...]

    # Connection Laplacian
    Lconn = computeConnectionLaplacian(pts, Lc, N, tangentBasis)        # 2n x 2n real

    # Construção do Operador
    t = 0.4
    I = np.array(np.eye((nopts)), dtype=np.complex128)     
    operator = buildVecHeatOperator(I, Lconn, t)        # I - t*Lconn

    # Construção do Y0
    Y0, normsAllSame, firstNorm = buildY0(sources, nopts)

    # Vector Heat Method
    dirInterpPacked = solveVectorHeat(operator, Y0)     # (I - t*Lconn) @ dirInterp = Y0

    ## Normalizar os vetores 
    dirInterp = np.zeros(nopts, dtype=np.complex128)
    for i in range(nopts):
        val = np.array([dirInterpPacked[2*i], dirInterpPacked[2*i+1]])      # dirInterpPacked (2n x 1) -> dirInterp (n x 1)

        ## Normalizar, com algum valor de corte para estabilidade
        norm = np.linalg.norm(val)
        if norm > 1e-10:
            dirInterp[i] = complex(val[0] / norm, val[1] / norm)
        else:
            dirInterp[i] = complex(1.0, 0.0)  # Valor padrão se o vetor for muito pequeno

    ## Definir a escala
    if normsAllSame:
        dirInterp *= firstNorm
    else:
        ## Interpolar magnitudes usando a equação de calor escalar
        rhsNorm = np.zeros(nopts)
        rhsOnes = np.zeros(nopts)
        
        ## Preencher os valores nos pontos de origem
        for idx, vec in sources:
            rhsOnes[idx] = 1.0
            rhsNorm[idx] = np.linalg.norm(vec)
        
        ## Construir o operador para calor escalar (M - tL)
        scalarOp = eye(nopts, format='csr') - t * csr_matrix(Lc)
        
        ## Resolver os sistemas para as normas e para os 1's
        interpNorm = spsolve(scalarOp, rhsNorm)
        interpOnes = spsolve(scalarOp, rhsOnes)
        
        ## Aplicar os fatores de escala aos vetores normalizados
        for i in range(nopts):
            if abs(interpOnes[i]) > 1e-10:
                scaleFactor = interpNorm[i] / interpOnes[i]
                dirInterp[i] *= scaleFactor

    result = np.zeros((nopts, 2))
    for i in range(nopts):
        result[i, 0] = dirInterp[i].real
        result[i, 1] = dirInterp[i].imag

    # edgess = build_sphere_of_influence_graph(pts)
    # edges = np.array(edgess)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Superficie", pts, tri, smooth_shade=True)
    ps_mesh.add_vector_quantity("Normal", N)
    # ps_mesh.add_vector_quantity("Tangente", T)  # axis-x
    # ps_mesh.add_vector_quantity("Binormal", B)  # axis-y
    ps_mesh.add_tangent_vector_quantity("Vetores de entrada", sourceVectors, T, B)
    ps_mesh.add_tangent_vector_quantity("Vector Heat", result, T, B)
    # ps_curve = ps.register_curve_network("Grafo SIG", pts, edges)
    # ps_mesh.add_tangent_vector_quantity("Transporte Paralelo", V, T, B)
    # ps.register_point_cloud("Ponto fonte", pts[0,:].reshape((1,-1)), radius=0.003, color=(1,0,0))

    ps.show()

if __name__ == '__main__': main()