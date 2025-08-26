# Advanced skull mesh refinement using specialized libraries
# You'll need to install these packages:
# pip install open3d pymeshlab trimesh pyvista scikit-image

import numpy as np
import open3d as o3d
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    print("PyMeshLab not available. Install with: pip install pymeshlab")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Trimesh not available. Install with: pip install trimesh")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")

class AdvancedSkullMeshRefinement:
    def __init__(self, coarse_points):
        """
        Advanced mesh refinement using specialized libraries
        
        Args:
            coarse_points: numpy array of shape (N, 3) with x, y, z coordinates
        """
        self.coarse_points = np.array(coarse_points)
        self.mesh = None
        self.fine_points = None
    
    def method_open3d_poisson(self, depth=8, density_threshold=0.1):
        """
        Method using Open3D's Poisson surface reconstruction
        This is often the best method for skull surfaces
        """
        print("Open3D Poisson Surface Reconstruction")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coarse_points)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Orient normals consistently (important for closed surfaces like skulls)
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low density vertices (noise)
        if density_threshold > 0:
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Sample points from the mesh surface
        self.mesh = mesh
        self.fine_points = np.asarray(mesh.vertices)
        
        return self.fine_points, mesh
    
    def method_open3d_ball_pivoting(self, radii=None):
        """
        Method using Open3D's Ball Pivoting Algorithm
        Good for preserving local features
        """
        print("Open3D Ball Pivoting Algorithm")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coarse_points)
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        # Estimate radii if not provided
        if radii is None:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
        
        # Ball pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        # Subdivide the mesh to increase density
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        
        self.mesh = mesh
        self.fine_points = np.asarray(mesh.vertices)
        
        return self.fine_points, mesh
    
    def method_open3d_alpha_shapes(self, alpha=None):
        """
        Method using Alpha Shapes
        Good for complex, non-convex surfaces
        """
        print("Open3D Alpha Shapes")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coarse_points)
        
        # Estimate alpha if not provided
        if alpha is None:
            distances = pcd.compute_nearest_neighbor_distance()
            alpha = np.mean(distances) * 2
        
        # Create alpha shape
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        # Clean up the mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Subdivide for higher density
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        
        self.mesh = mesh
        self.fine_points = np.asarray(mesh.vertices)
        
        return self.fine_points, mesh
    
    def method_pymeshlab_refinement(self, target_face_count=5000):
        """
        Method using PyMeshLab for advanced mesh processing
        """
        if not PYMESHLAB_AVAILABLE:
            print("PyMeshLab not available")
            return None, None
            
        print("PyMeshLab Advanced Refinement")
        
        # Create MeshSet
        ms = pymeshlab.MeshSet()
        
        # First, create a basic mesh from points using Delaunay
        from scipy.spatial import Delaunay
        
        # Project to 2D for triangulation (simplified approach)
        # For a better approach, use Poisson reconstruction in PyMeshLab
        centered_points = self.coarse_points - np.mean(self.coarse_points, axis=0)
        _, _, vh = np.linalg.svd(centered_points)
        proj_2d = centered_points @ vh[:2].T
        tri = Delaunay(proj_2d)
        
        # Create mesh
        faces = tri.simplices
        mesh = pymeshlab.Mesh(self.coarse_points, faces)
        ms.add_mesh(mesh)
        
        # Apply various MeshLab filters
        # 1. Surface reconstruction (Poisson)
        ms.generate_surface_reconstruction_screened_poisson(depth=8)
        
        # 2. Remesh to target face count
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PercentageValue(1.0),
            adaptive=True,
            selectedonly=False
        )
        
        # 3. Smooth the mesh
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)
        
        # 4. Subdivide if needed
        current_face_count = ms.current_mesh().face_number()
        if current_face_count < target_face_count:
            iterations = int(np.log2(target_face_count / current_face_count))
            ms.meshing_surface_subdivision_midpoint(iterations=max(1, iterations))
        
        # Extract refined mesh
        refined_mesh = ms.current_mesh()
        self.fine_points = refined_mesh.vertex_matrix()
        
        return self.fine_points, refined_mesh
    
    def method_trimesh_refinement(self, subdivision_iterations=2):
        """
        Method using Trimesh library
        """
        if not TRIMESH_AVAILABLE:
            print("Trimesh not available")
            return None, None
            
        print("Trimesh Refinement")
        
        # Create initial mesh using convex hull
        try:
            mesh = trimesh.convex.convex_hull(self.coarse_points)
            
            # Subdivide the mesh
            for _ in range(subdivision_iterations):
                mesh = mesh.subdivide()
            
            # Project vertices to closest original points
            for i, vertex in enumerate(mesh.vertices):
                distances = np.linalg.norm(self.coarse_points - vertex, axis=1)
                closest_idx = np.argmin(distances)
                # Blend between original position and closest point
                blend_factor = 0.7
                mesh.vertices[i] = (blend_factor * self.coarse_points[closest_idx] + 
                                  (1 - blend_factor) * vertex)
            
            # Smooth the mesh
            mesh = mesh.smoothed()
            
            self.mesh = mesh
            self.fine_points = mesh.vertices
            
            return self.fine_points, mesh
            
        except Exception as e:
            print(f"Trimesh refinement failed: {e}")
            return None, None
    
    def method_pyvista_refinement(self, subdivisions=2):
        """
        Method using PyVista for mesh processing
        """
        if not PYVISTA_AVAILABLE:
            print("PyVista not available")
            return None, None
            
        print("PyVista Refinement")
        
        # Create point cloud
        cloud = pv.PolyData(self.coarse_points)
        
        # Surface reconstruction using Delaunay 3D
        try:
            # First create a convex hull
            hull = cloud.delaunay_3d().extract_surface()
            
            # Subdivide
            for _ in range(subdivisions):
                hull = hull.subdivide(subfilter='linear')
            
            # Smooth
            smoothed = hull.smooth(n_iter=50, relaxation_factor=0.1)
            
            self.mesh = smoothed
            self.fine_points = smoothed.points
            
            return self.fine_points, smoothed
            
        except Exception as e:
            print(f"PyVista refinement failed: {e}")
            return None, None
    
    def sample_surface_points(self, n_points=5000):
        """
        Sample additional points from the refined mesh surface
        """
        if self.mesh is None:
            print("No mesh available. Run a refinement method first.")
            return None
            
        if hasattr(self.mesh, 'sample_points_uniformly'):  # Open3D mesh
            sampled_pcd = self.mesh.sample_points_uniformly(number_of_points=n_points)
            return np.asarray(sampled_pcd.points)
        elif hasattr(self.mesh, 'sample'):  # Trimesh
            sampled_points, _ = self.mesh.sample(n_points)
            return sampled_points
        elif hasattr(self.mesh, 'sample'):  # PyVista
            sampled = self.mesh.sample(n_points)
            return sampled.points
        else:
            print("Mesh type not supported for sampling")
            return None
    
    def visualize_with_open3d(self):
        """
        Interactive visualization using Open3D
        """
        # Original points
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(self.coarse_points)
        pcd_orig.paint_uniform_color([1, 0, 0])  # Red
        
        vis_items = [pcd_orig]
        
        # Fine points
        if self.fine_points is not None:
            pcd_fine = o3d.geometry.PointCloud()
            pcd_fine.points = o3d.utility.Vector3dVector(self.fine_points)
            pcd_fine.paint_uniform_color([0, 0, 1])  # Blue
            vis_items.append(pcd_fine)
        
        # Mesh
        if self.mesh is not None and hasattr(self.mesh, 'triangles'):
            mesh_vis = self.mesh.copy()
            mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
            mesh_vis.compute_vertex_normals()
            vis_items.append(mesh_vis)
        
        # Visualize
        o3d.visualization.draw_geometries(vis_items)
    
    def save_mesh(self, filename):
        """
        Save the refined mesh to file
        """
        if self.mesh is None:
            print("No mesh to save")
            return False
            
        try:
            if hasattr(self.mesh, 'export'):  # Trimesh
                self.mesh.export(filename)
            elif hasattr(self.mesh, 'save'):  # PyVista
                self.mesh.save(filename)
            else:  # Open3D
                o3d.io.write_triangle_mesh(filename, self.mesh)
            print(f"Mesh saved to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save mesh: {e}")
            return False


# Example usage
def demonstrate_advanced_methods():
    """
    Demonstrate the advanced methods
    """
    # Create example skull data
    def create_skull_data():
        n_points = 150
        a, b, c = 8, 6, 7
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        noise = 1 + 0.3 * (np.random.random(n_points) - 0.5)
        
        x = a * np.sin(phi) * np.cos(theta) * noise
        y = b * np.sin(phi) * np.sin(theta) * noise
        z = c * np.cos(phi) * noise
        
        return np.column_stack([x, y, z])
    
    coarse_points = create_skull_data()
    refiner = AdvancedSkullMeshRefinement(coarse_points)
    
    print("Testing advanced methods:\n")
    
    # Test Open3D Poisson (usually the best for skull surfaces)
    try:
        fine_points, mesh = refiner.method_open3d_poisson(depth=7)
        print(f"Poisson: {len(coarse_points)} → {len(fine_points)} points")
        
        # Sample even more points from the surface
        sampled_points = refiner.sample_surface_points(n_points=3000)
        if sampled_points is not None:
            print(f"Sampled additional {len(sampled_points)} surface points")
        
        # Interactive visualization
        print("Opening interactive visualization...")
        refiner.visualize_with_open3d()
        
    except Exception as e:
        print(f"Open3D Poisson failed: {e}")
    
    # Test other methods
    methods = [
        ('Ball Pivoting', refiner.method_open3d_ball_pivoting),
        ('Alpha Shapes', refiner.method_open3d_alpha_shapes),
    ]
    
    if PYMESHLAB_AVAILABLE:
        methods.append(('PyMeshLab', refiner.method_pymeshlab_refinement))
    
    if TRIMESH_AVAILABLE:
        methods.append(('Trimesh', refiner.method_trimesh_refinement))
    
    if PYVISTA_AVAILABLE:
        methods.append(('PyVista', refiner.method_pyvista_refinement))
    
    for name, method in methods:
        try:
            fine_points, mesh = method()
            if fine_points is not None:
                print(f"{name}: {len(coarse_points)} → {len(fine_points)} points")
        except Exception as e:
            print(f"{name} failed: {e}")


if __name__ == "__main__":
    demonstrate_advanced_methods()