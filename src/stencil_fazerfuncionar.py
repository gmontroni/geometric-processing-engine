import numpy as np
import meshio, sys, os
import polyscope as ps
from scipy.spatial import KDTree

from geopackages.vet.pyvet import VET
from geopackages.rbf.rbf_fd_operators import rbf_fd_weights 

def main():

    # Input mesh
    # Verificar se um arquivo foi especificado
    if len(sys.argv) <= 1:
        print("Erro: É necessário especificar um arquivo de malha.")
        print("Uso: uv run src/stencil.py <arquivo.obj>")
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

    # source = 11678
    # source = 7599
    # source = 7716
    source = [4102,3330,2292,3698,3945,2275,4098]       # bunny
    # source = [11678,7599,7716,8365]         # knot

    print('Construindo o estêncil...')
    # Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators(pts, T, B)
    # Gx3D, Gy3D, Gz3D, Lc = compute_surface_operators3dd(pts, T, B, N)
    # Gx3D, Gy3D, Gz3D, Lc, vecIdx = compute_surface_operators_with_reliability(pts, T, B, N)

    ## Score
    tree = KDTree(pts)

    # Initialize arrays to store scores and indices
    k = 50 # Number of neighbors to consider
    scores = np.zeros((nopts, k))
    indices = np.zeros((nopts, k), dtype=int)
    outliers = np.zeros((nopts, k), dtype=float)
    for i in range(nopts):
        # Get k nearest neighbors
        distances, idx = tree.query(pts[i,:], k=k+1)  # +1 because the point itself is included
        
        # Skip the point itself (idx[0])
        idx = idx[1:]
        distances = distances[1:]
        
        # Store indices
        indices[i] = idx
        
        # Compute radius as distance to k-th neighbor (for normalization)
        r = distances[-1]
        
        # Normal at current point
        n_p = N[i]

        theta_max = np.pi  # Maximum angle for normalization
        limit_theta_max = np.degrees(theta_max)
        w_theta = 0.34
        w_proj = 0.33
        w_dist = 0.33
        for j, neighbor_idx in enumerate(idx):
            # Point q (neighbor)
            q = pts[neighbor_idx]
            
            # Normal at neighbor point
            n_q = N[neighbor_idx]
            
            # Angle between normals: θ(p, q) = arccos(n_p · n_q)
            cos_theta = np.clip(np.dot(n_p, n_q), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            # Vector from p to q
            p_to_q = q - pts[i]
            
            # Distance from q to tangent plane at p: d_⊥ = |(q - p) · n_p|
            d_perp = abs(np.dot(p_to_q, n_p))
            
            # Projection of p_to_q onto tangent plane
            proj_vector = p_to_q - d_perp * n_p
            d_parallel = np.linalg.norm(proj_vector)
            
            # Euclidean distance between p and q
            distance = distances[j]
            
            # Compute the score components
            theta_term = (theta / theta_max) ** 2
            proj_term = (d_perp / ( d_parallel + 1e-6)) ** 2
            dist_term = (distance / r) ** 2
            
            # Compute the final score
            score = w_theta * theta_term + w_proj * proj_term + w_dist * dist_term
            
            # Store the score
            scores[i, j] = score

            # outliers[i, j] = 1 if np.dot(p_to_q, n_p) > 0 else 0
            outliers[i, j] = np.dot(n_p, n_q)

    max_neighbors = 25
    vecindices = np.zeros((nopts, max_neighbors,1), dtype=int)

    for i in range(nopts):
        # Get k nearest neighbors
        _, idx_full = tree.query(pts[i,:], k=k+1)
        idx_full = idx_full[1:]  # Exclude the point itself
        
        # Get scores for these neighbors
        neighbor_scores = scores[i, :len(idx_full)]
        
        # Select the max_neighbors with the lowest scores (most reliable)
        best_indices = np.argsort(neighbor_scores)[:max_neighbors]
        idx = idx_full[best_indices]
        vecindices[i, :len(idx), 0] = idx

    print('Construindo os operadores via RBF-FD...')
    ### Calculo dos operadores
    Lc    = np.zeros((nopts, nopts))
    Gx3D  = np.zeros((nopts, nopts))
    Gy3D  = np.zeros((nopts, nopts))
    Gz3D  = np.zeros((nopts, nopts))
    for i in range(nopts):
        # # Get k nearest neighbors
        # _, idx_full = tree.query(pts[i,:], k=k+1)
        # idx_full = idx_full[1:]  # Exclude the point itself
        
        # # Get scores for these neighbors
        # neighbor_scores = scores[i, :len(idx_full)]
        
        # # Select the max_neighbors with the lowest scores (most reliable)
        # best_indices = np.argsort(neighbor_scores)[:max_neighbors]
        # idx = idx_full[best_indices]
        idx = vecindices[i,:,:].reshape(vecindices.shape[1],).tolist()

        # Compute operators using the selected reliable neighbors
        R = np.hstack((T[i,:].reshape(3,1), B[i,:].reshape(3,1)))
        Xloc = R.T @ (pts[i,:] - pts[idx,:]).T      
        Xloc = Xloc.T
        W = rbf_fd_weights(Xloc, np.array([0, 0]), 5, 5)
        Lc[i, idx] = W[:,0]

        temp = R @ W[:,1:3].T
        Gx3D[i,idx] = temp[0,:]
        Gy3D[i,idx] = temp[1,:]
        Gz3D[i,idx] = temp[2,:]

    ### Aplicação dos operadores
    source_function = np.zeros(nopts)
    source = 40               ## Ponto fonte (centro paraboloide)
    # source = [source] + mesh.compute_k_ring(source,1)

    lap3D = Gx3D @ Gx3D + Gy3D @ Gy3D + Gz3D @ Gz3D       # Divergente do Gradiente 3D

    # Copy para Dirichlet
    lap3D_heat = lap3D.copy()
    Lc_heat = Lc.copy()

    source_heat = source_function.copy()
    source_heat[source] = 1
    # source_heat = source_heat / np.max(source_heat)

    t = 0.01
    phi_lap = source_heat.copy()
    phi_Lc = source_heat.copy()
    # phi_lap = np.linalg.solve(np.eye(nopts)-t*lap3D_heat, phi_lap)
    phi_lap = np.linalg.solve(np.eye(nopts)-t*lap3D_heat, phi_lap)
    phi_Lc = np.linalg.solve(np.eye(nopts)-t*Lc_heat, phi_Lc)
    for i in range(1,3):
        phi_lap = np.linalg.solve(np.eye(nopts)-t*lap3D_heat, phi_lap)
        phi_Lc = np.linalg.solve(np.eye(nopts)-t*Lc_heat, phi_Lc)


    # Draw
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    ps_mesh.add_scalar_quantity("Função", source_heat, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação do Calor", phi_lap, cmap='turbo')
    ps_mesh.add_scalar_quantity("Equação do Calor Lc", phi_Lc, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Função", u, cmap='turbo')
    # ps_mesh.add_scalar_quantity("f0 eq Calor", f0, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Solução eq Calor", f, cmap='turbo')
    # ps_mesh.add_vector_quantity("Gradiente Exato", exactGradient)
    # ps_mesh.add_vector_quantity("Gradiente", gradient)
    # ps_mesh.add_scalar_quantity("Divergente do Gradiente", div, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Laplaciano Exato", exactLapla, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Laplaciano", Lapla, cmap='turbo')
    # ps_mesh.add_scalar_quantity("Erro do Gradiente", graderror, cmap='turbo')
    # ps.register_point_cloud("Ponto fonte", pts[source,:].reshape((1,-1)), radius=0.003, color=(0,0,0))
    ps.register_point_cloud("Ponto fonte", pts[source,:], radius=0.003, color=(0,0,0))
    ps.register_point_cloud("Vizinhos do ponto fonte", pts[vecindices[source,:,:],:].reshape((-1,3)), radius=0.003, color=(1,0,0))

    ps.show()

if __name__ == '__main__': main()