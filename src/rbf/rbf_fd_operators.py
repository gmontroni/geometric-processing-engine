import numpy as np
from scipy.spatial import KDTree
from rbf.weights_rbf_fd_2d import rbf_fd_weights, rbf_fd_weights_derivatives_only

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

def compute_surface_operators3d(pts, T, B, N):
    # pts: n x 3
    nopts = pts.shape[0]

    # KD-Tree
    tree = KDTree(pts)

    dotProduct = np.dot(N, N.T)
    cosTheta = 1 - np.clip(dotProduct, -1.0, 1.0)

    Lc    = np.zeros((nopts, nopts))
    Gx3D  = np.zeros((nopts, nopts))
    Gy3D  = np.zeros((nopts, nopts))
    Gz3D  = np.zeros((nopts, nopts))
    for i in range(nopts):
        
        _, idx = tree.query(pts[i,:], k=50)
        sortedIdx = np.argsort(cosTheta[i, idx])
        idxnormals = sortedIdx[:30]

        R = np.hstack((T[i,:].reshape(3,1), B[i,:].reshape(3,1)))
        Xloc = R.T @ (pts[i,:] - pts[idx[idxnormals],:]).T      
        Xloc = Xloc.T
        W = rbf_fd_weights(Xloc, np.array([0, 0]), 5, 5)        # 5 5
        Lc[i, idx[idxnormals]] = W[:,0]

        temp = R[:, :2] @ W[:,1:3].T
        Gx3D[i,idx[idxnormals]] = temp[0,:]
        Gy3D[i,idx[idxnormals]] = temp[1,:]
        Gz3D[i,idx[idxnormals]] = temp[2,:]

    return Gx3D, Gy3D, Gz3D, Lc

def compute_surface_operators(pts, T, B):
    # pts: n x 3
    nopts = pts.shape[0]

    # KD-Tree
    tree = KDTree(pts)

    Lc    = np.zeros((nopts, nopts))
    Gx3D  = np.zeros((nopts, nopts))
    Gy3D  = np.zeros((nopts, nopts))
    Gz3D  = np.zeros((nopts, nopts))
    for i in range(nopts):
        
        _, idx = tree.query(pts[i,:], k=30)

        R = np.hstack((T[i,:].reshape(3,1), B[i,:].reshape(3,1)))
        Xloc = R.T @ (pts[i,:] - pts[idx,:]).T      
        Xloc = Xloc.T
        W = rbf_fd_weights(Xloc, np.array([0, 0]), 5, 5)        # 5 5
        Lc[i, idx] = W[:,0]

        temp = R @ W[:,1:3].T
        Gx3D[i,idx] = temp[0,:]
        Gy3D[i,idx] = temp[1,:]
        Gz3D[i,idx] = temp[2,:]

    return Gx3D, Gy3D, Gz3D, Lc

def compute_surface_operators_score(pts, N, k=50, w_theta=0.34, w_proj=0.33, w_dist=0.33, 
                                   theta_max=np.pi, epsilon=1e-6):
    """
    Compute a connectivity score between points in a point cloud.
    
    Lower scores indicate higher confidence that neighbors are topologically consistent.
    The score penalizes:
    - Normal misalignment (large θ)
    - Deviations from the tangent plane (high d_⊥/d_∥)
    - Excessive spatial distance (high ||p-q||)
    
    Args:
        pts: array of shape (n, 3) containing 3D points
        N: array of shape (n, 3) containing surface normals at each point
        k: number of nearest neighbors to consider
        w_theta: weight for normal angle term
        w_proj: weight for projection term
        w_dist: weight for distance term
        theta_max: maximum tolerable angle between normals in radians (default: 120 degrees)
        epsilon: small constant to avoid division by zero
        
    Returns:
        scores: array of shape (n, k) containing connectivity scores for each point and its k neighbors
        indices: array of shape (n, k) containing indices of the k neighbors for each point
    """
    nopts = pts.shape[0]
    
    # Verify weights sum to 1
    assert abs(w_theta + w_proj + w_dist - 1.0) < 1e-6, "Weights must sum to 1"
    
    # KD-Tree for nearest neighbor search
    tree = KDTree(pts)

    # Initialize arrays to store scores and indices
    scores = np.zeros((nopts, k))
    indices = np.zeros((nopts, k), dtype=int)
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
            proj_term = (d_perp / (d_parallel + 1e-6)) ** 2
            dist_term = (distance / r) ** 2
            
            # Compute the final score
            score = w_theta * theta_term + w_proj * proj_term + w_dist * dist_term
            
            # Store the score
            scores[i, j] = score
    
    return scores, indices

def compute_surface_operators_with_reliability(pts, T, B, N, k=50, max_neighbors=30, 
                                             w_theta=0.34, w_proj=0.33, w_dist=0.33):
    """
    Compute surface operators using the reliability score to select the most reliable neighbors.
    
    Args:
        pts: array of shape (n, 3) containing 3D points
        T: array of shape (n, 3) containing tangent vectors
        B: array of shape (n, 3) containing bitangent vectors
        N: array of shape (n, 3) containing normal vectors
        k: number of potential neighbors to consider initially
        max_neighbors: maximum number of neighbors to use for operator computation
        w_theta, w_proj, w_dist: weights for score components
        
    Returns:
        Gx3D, Gy3D, Gz3D, Lc: surface operators
    """
    nopts = pts.shape[0]
    
    # Compute connectivity scores for neighbors
    scores, _ = compute_surface_operators_score(pts, N, k=k, 
                                             w_theta=w_theta, w_proj=w_proj, w_dist=w_dist)
    
    # KD-Tree
    tree = KDTree(pts)
    
    Lc    = np.zeros((nopts, nopts))
    Gx3D  = np.zeros((nopts, nopts))
    Gy3D  = np.zeros((nopts, nopts))
    Gz3D  = np.zeros((nopts, nopts))
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
        vecindices[i, :len(idx), 0] = idx
    
    return Gx3D, Gy3D, Gz3D, Lc, vecindices