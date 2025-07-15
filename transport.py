import numpy as np
import meshio, sys, os
import polyscope as ps

sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'src'))
from pyvet import VET
from parallelTransport import parallelTransport
from objtovtk import addVelocities_obj2vtk

def main():

    # Input mesh
    # Verificar se um arquivo foi especificado
    if len(sys.argv) <= 1:
        print("Erro: É necessário especificar um arquivo de malha.")
        print("Uso: python3 vectorHeat.py <arquivo.obj>")
        sys.exit(1)
        
    meshName = sys.argv[1]
    if '/' not in meshName:
        fileName = f'input/{meshName}'
    else:
        fileName = meshName
    # print(f"Carregando malha: {meshName}")

    mesh = meshio.read(fileName, file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts, _ = pts.shape[0], tri.shape[0]

    # Source
    # source = 2929

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    # rings = vertexRings(mesh, pts.shape[0])
    T, B, N = mesh.computeOrthonormalBase()             # T e B criados usando normal
    del mesh

    # Transporte Paralelo
    _, V = parallelTransport(pts,T,B,N,[1,0,0])         # usar sem a conectividade (não tem diferença)

    # vtk version
    addVelocities_obj2vtk(fileName, V)

    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Superficie", pts, tri, smooth_shade=True)
    # ps_mesh.add_vector_quantity("Normal", N)
    ps_mesh.add_vector_quantity("Tangente", T)  # axis-x
    ps_mesh.add_vector_quantity("Binormal", B)  # axis-y
    ps_mesh.add_vector_quantity("Vetor Transportado", V)
    # ps.register_point_cloud("Ponto Fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(1,0,0))

    ps.show()

if __name__ == '__main__': main()