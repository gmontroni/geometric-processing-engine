import numpy as np
import meshio, sys, os
import polyscope as ps
from scipy.spatial import KDTree

from mesh.pyvet import VET

# Global variables for interactive parameters
w_theta = 0.34
w_proj = 0.33
w_dist = 0.33
current_source_idx = 0
max_neighbors = 25  # Add this global variable

# Global data
pts = None
tri = None
N = None
tree = None
source = None
vecindices = None
ps_mesh = None

def compute_scores_for_weights(w_t, w_p, w_d):
    """Compute scores for all points with given weights"""
    nopts = pts.shape[0]
    k = 50
    scores = np.zeros((nopts, k))
    
    for i in range(nopts):
        distances, idx = tree.query(pts[i,:], k=k+1)
        idx = idx[1:]
        distances = distances[1:]
        
        r = distances[-1]
        n_p = N[i]
        theta_max = np.pi
        
        for j, neighbor_idx in enumerate(idx):
            q = pts[neighbor_idx]
            n_q = N[neighbor_idx]
            
            cos_theta = np.clip(np.dot(n_p, n_q), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            p_to_q = q - pts[i]
            d_perp = abs(np.dot(p_to_q, n_p))
            proj_vector = p_to_q - d_perp * n_p
            d_parallel = np.linalg.norm(proj_vector)
            distance = distances[j]
            
            theta_term = (theta / theta_max) ** 2
            proj_term = (d_perp / (d_parallel + 1e-6)) ** 2
            dist_term = (distance / r) ** 2
            
            score = w_t * theta_term + w_p * proj_term + w_d * dist_term
            scores[i, j] = score
    
    return scores

def update_visualization():
    """Update the visualization with current parameters"""
    global w_theta, w_proj, w_dist, max_neighbors
    
    # Compute new scores
    scores = compute_scores_for_weights(w_theta, w_proj, w_dist)
    
    # Update vecindices for current weights
    new_vecindices = np.zeros((pts.shape[0], max_neighbors, 1), dtype=int)
    
    for i in range(pts.shape[0]):
        _, idx_full = tree.query(pts[i,:], k=50+1)
        idx_full = idx_full[1:]
        
        neighbor_scores = scores[i, :len(idx_full)]
        best_indices = np.argsort(neighbor_scores)[:max_neighbors]
        idx = idx_full[best_indices]
        
        # Handle case where we have fewer neighbors than max_neighbors
        actual_neighbors = min(len(idx), max_neighbors)
        new_vecindices[i, :actual_neighbors, 0] = idx[:actual_neighbors]
    
    # Remove existing point clouds
    ps.remove_point_cloud("All Source Points", error_if_absent=False)
    for i in range(len(source)):
        ps.remove_point_cloud(f"Source {i} Neighbors", error_if_absent=False)
    
    # Add all source points
    all_source_pts = pts[source,:].reshape((-1,3))
    ps.register_point_cloud("All Source Points", all_source_pts, radius=0.004, color=(0,0,0))
    
    # Add neighbors for each source point with different colors
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (0.5,0.5,0.5), (1,0.5,0)]
    
    # Create combined scalar field for all source influences
    combined_score_field = np.zeros(pts.shape[0])
    
    for i, current_source in enumerate(source):
        # Get neighbors for this source
        source_neighbors = new_vecindices[current_source, :, 0]
        valid_neighbors = source_neighbors[source_neighbors > 0]
        
        if len(valid_neighbors) > 0:
            neighbor_pts = pts[valid_neighbors,:].reshape((-1,3))
            color = colors[i % len(colors)]
            ps.register_point_cloud(f"Source {i} Neighbors", neighbor_pts, radius=0.004, color=color)
            
            # Add to combined score field
            neighbor_scores = scores[current_source, :]
            inverted_scores = 1.0 - neighbor_scores[:len(valid_neighbors)]
            
            # Use maximum influence if multiple sources affect same point
            for j, neighbor_idx in enumerate(valid_neighbors):
                combined_score_field[neighbor_idx] = max(combined_score_field[neighbor_idx], inverted_scores[j])
    
    # Add combined scalar quantity
    ps_mesh.add_scalar_quantity("Combined Neighbor Influence", combined_score_field, cmap='viridis')

def callback():
    """Polyscope callback for interactive controls"""
    global w_theta, w_proj, w_dist, max_neighbors
    
    # Weight sliders (without auto-update)
    _, w_theta = ps.imgui.SliderFloat("w_theta", w_theta, v_min=0.0, v_max=1.0)
    _, w_proj = ps.imgui.SliderFloat("w_proj", w_proj, v_min=0.0, v_max=1.0)
    _, w_dist = ps.imgui.SliderFloat("w_dist", w_dist, v_min=0.0, v_max=1.0)
    
    # Normalize weights in real-time for display
    total = w_theta + w_proj + w_dist
    if total > 0:
        norm_w_theta = w_theta / total
        norm_w_proj = w_proj / total
        norm_w_dist = w_dist / total
    else:
        norm_w_theta = norm_w_proj = norm_w_dist = 1.0/3.0
    
    # Max neighbors slider
    _, max_neighbors = ps.imgui.SliderInt("Max Neighbors", max_neighbors, v_min=5, v_max=50)
    
    # Separator for better organization
    ps.imgui.Separator()
    
    # Quick preset buttons
    if ps.imgui.Button("Equal Weights"):
        w_theta = w_proj = w_dist = 1.0/3.0
    
    ps.imgui.SameLine()
    if ps.imgui.Button("Angle Focus"):
        w_theta, w_proj, w_dist = 0.7, 0.2, 0.1
    
    ps.imgui.SameLine()
    if ps.imgui.Button("Distance Focus"):
        w_theta, w_proj, w_dist = 0.1, 0.2, 0.7
    
    ps.imgui.SameLine()
    if ps.imgui.Button("Projection Focus"):
        w_theta, w_proj, w_dist = 0.1, 0.7, 0.2
    
    ps.imgui.Separator()
    
    # Main update button (bigger and more prominent)
    if ps.imgui.Button("Update Visualization", size=(200, 40)):
        # Update global normalized weights
        w_theta = norm_w_theta
        w_proj = norm_w_proj
        w_dist = norm_w_dist
        update_visualization()
    
    ps.imgui.Separator()
    
    # Display current normalized weights (preview)
    ps.imgui.Text("Preview of normalized weights:")
    ps.imgui.Text(f"  θ = {norm_w_theta:.3f}")
    ps.imgui.Text(f"  proj = {norm_w_proj:.3f}")
    ps.imgui.Text(f"  dist = {norm_w_dist:.3f}")
    ps.imgui.Text(f"Max neighbors: {max_neighbors}")
    ps.imgui.Text(f"Total source points: {len(source)}")
    
    # List all source points
    ps.imgui.Text("Source points:")
    for i, src in enumerate(source):
        color = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (0.5,0.5,0.5), (1,0.5,0)][i % 8]
        ps.imgui.TextColored((*color, 1.0), f"  {i}: {src}")
    
    ps.imgui.Separator()
    ps.imgui.TextColored((0.7, 0.7, 0.7, 1.0), "Adjust parameters above, then click UPDATE")

def main():
    global pts, tri, N, tree, source, vecindices, ps_mesh, max_neighbors

    # Input mesh
    if len(sys.argv) <= 1:
        print("Erro: É necessário especificar um arquivo de malha.")
        print("Uso: uv run src/stencil_visualization.py <arquivo.obj>")
        sys.exit(1)
        
    meshName = sys.argv[1]
    if '/' not in meshName:
        fileName = f'meshes/{meshName}'
    else:
        fileName = meshName

    mesh = meshio.read(fileName, file_format='obj')
    pts = mesh.points
    tri = np.array(mesh.cells_dict['triangle'])
    nopts = pts.shape[0]

    # Construção das coordenadas locais
    mesh = VET(pts, tri)
    T, B, N = mesh.computeOrthonormalBase()
    del mesh

    source = [4102,3330,2292,3698,3945,2275,4098]       # bunny
    # source = [11678,7599,7716,8365]  # knot
    
    # Make sure source indices are valid
    source = [s for s in source if s < nopts]
    if not source:
        source = [0]  # fallback
    
    print('Construindo o estêncil...')
    tree = KDTree(pts)

    # Initial computation
    scores = compute_scores_for_weights(w_theta, w_proj, w_dist)
    
    # Initial vecindices computation
    vecindices = np.zeros((nopts, max_neighbors, 1), dtype=int)
    
    for i in range(nopts):
        _, idx_full = tree.query(pts[i,:], k=50+1)
        idx_full = idx_full[1:]
        
        neighbor_scores = scores[i, :len(idx_full)]
        best_indices = np.argsort(neighbor_scores)[:max_neighbors]
        idx = idx_full[best_indices]
        
        actual_neighbors = min(len(idx), max_neighbors)
        vecindices[i, :actual_neighbors, 0] = idx[:actual_neighbors]

    # Initialize Polyscope
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    
    # Register callback
    ps.set_user_callback(callback)
    
    # Register mesh
    ps_mesh = ps.register_surface_mesh("Mesh", pts, tri, smooth_shade=True)
    
    # Initial visualization
    update_visualization()
    
    ps.show()

if __name__ == '__main__': main()