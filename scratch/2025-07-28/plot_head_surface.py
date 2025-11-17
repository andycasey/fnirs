import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import matplotlib.tri as mtri

def create_3d_head_mesh(x_data, y_data, z_data, values, resolution=50):
    """
    Create a 3D mesh visualization of head position data with color interpolation.
    
    Parameters:
    - x_data, y_data, z_data: arrays of 3D coordinates
    - values: array of values at each (x,y,z) point for coloring
    - resolution: resolution of the interpolated mesh
    """
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Method 1: 3D Scatter plot with interpolated surface projections
    def plot_with_projections():
        # Plot the original data points
        scatter = ax.scatter(x_data, y_data, z_data, c=values, 
                           cmap='viridis', s=50, alpha=0.8)
        
        # Create 2D projections and interpolate
        # XY plane projection
        xi = np.linspace(x_data.min(), x_data.max(), resolution)
        yi = np.linspace(y_data.min(), y_data.max(), resolution)
        XI, YI = np.meshgrid(xi, yi)
        
        # Interpolate Z values for XY plane
        ZI = griddata((x_data, y_data), z_data, (XI, YI), method='cubic', fill_value=z_data.min())
        VI = griddata((x_data, y_data), values, (XI, YI), method='cubic', fill_value=values.min())
        
        # Plot the interpolated surface
        surface = ax.plot_surface(XI, YI, ZI, facecolors=plt.cm.viridis(VI/VI.max()), 
                                alpha=0.6, antialiased=True)
        
        return scatter
    
    # Method 2: Triangulated mesh (alternative approach)
    def plot_triangulated_mesh():
        # Create 2D triangulation in XY plane
        points_2d = np.column_stack((x_data, y_data))
        tri = Delaunay(points_2d)
        
        # Create triangulated surface
        triangles = tri.simplices
        
        # Plot triangulated surface with color interpolation
        for triangle in triangles:
            triangle_points = np.array([
                [x_data[triangle[0]], y_data[triangle[0]], z_data[triangle[0]]],
                [x_data[triangle[1]], y_data[triangle[1]], z_data[triangle[1]]],
                [x_data[triangle[2]], y_data[triangle[2]], z_data[triangle[2]]]
            ])
            
            triangle_values = np.array([values[triangle[0]], values[triangle[1]], values[triangle[2]]])
            mean_value = np.mean(triangle_values)
            
            # Plot triangle
            ax.plot_trisurf([triangle_points[i, 0] for i in range(3)],
                           [triangle_points[i, 1] for i in range(3)],
                           [triangle_points[i, 2] for i in range(3)],
                           color=plt.cm.viridis(mean_value/values.max()),
                           alpha=0.7)
        
        # Plot original points
        scatter = ax.scatter(x_data, y_data, z_data, c=values, 
                           cmap='viridis', s=50, alpha=0.9)
        return scatter
    
    # Use the projection method (generally works better for scattered 3D data)
    scatter = plot_with_projections()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Customize the plot
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Head Position Mesh with Value Interpolation')
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([x_data.max()-x_data.min(), 
                         y_data.max()-y_data.min(), 
                         z_data.max()-z_data.min()]).max() / 2.0
    mid_x = (x_data.max()+x_data.min()) * 0.5
    mid_y = (y_data.max()+y_data.min()) * 0.5
    mid_z = (z_data.max()+z_data.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig, ax

def create_advanced_3d_mesh(x_data, y_data, z_data, values):
    """
    Advanced 3D mesh using triangulation in 3D space
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Method 1: Project to XY plane and create surface
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Create grid for interpolation
    resolution = 30
    xi = np.linspace(x_data.min(), x_data.max(), resolution)
    yi = np.linspace(y_data.min(), y_data.max(), resolution)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate Z and values
    ZI = griddata((x_data, y_data), z_data, (XI, YI), method='linear')
    VI = griddata((x_data, y_data), values, (XI, YI), method='linear')
    
    # Remove NaN values
    mask = ~np.isnan(ZI) & ~np.isnan(VI)
    
    # Plot surface
    surface1 = ax1.plot_surface(XI, YI, ZI, facecolors=plt.cm.viridis(VI/np.nanmax(VI)), 
                               alpha=0.7, antialiased=True)
    ax1.scatter(x_data, y_data, z_data, c=values, cmap='viridis', s=30)
    ax1.set_title('XY Projection Method')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # Method 2: Project to XZ plane
    ax2 = fig.add_subplot(132, projection='3d')
    zi = np.linspace(z_data.min(), z_data.max(), resolution)
    XI, ZI = np.meshgrid(xi, zi)
    YI = griddata((x_data, z_data), y_data, (XI, ZI), method='linear')
    VI = griddata((x_data, z_data), values, (XI, ZI), method='linear')
    
    surface2 = ax2.plot_surface(XI, YI, ZI, facecolors=plt.cm.viridis(VI/np.nanmax(VI)), 
                               alpha=0.7, antialiased=True)
    ax2.scatter(x_data, y_data, z_data, c=values, cmap='viridis', s=30)
    ax2.set_title('XZ Projection Method')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # Method 3: Convex hull approach
    ax3 = fig.add_subplot(133, projection='3d')
    from scipy.spatial import ConvexHull
    
    points = np.column_stack((x_data, y_data, z_data))
    try:
        hull = ConvexHull(points)
        
        # Plot the convex hull faces
        for simplex in hull.simplices:
            triangle_values = values[simplex]
            mean_value = np.mean(triangle_values)
            
            triangle_points = points[simplex]
            
            # Create triangle
            triangle = plt.Polygon(triangle_points[:, :2], 
                                 color=plt.cm.viridis(mean_value/values.max()))
            
            # Plot 3D triangle
            xs, ys, zs = triangle_points.T
            ax3.plot_trisurf(xs, ys, zs, color=plt.cm.viridis(mean_value/values.max()), 
                            alpha=0.7)
    except:
        # Fallback to scatter plot if convex hull fails
        ax3.scatter(x_data, y_data, z_data, c=values, cmap='viridis', s=50)
    
    ax3.set_title('Convex Hull Method')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    
    plt.tight_layout()
    return fig

# Example usage with sample data
def generate_sample_head_data():
    """Generate sample head position data for demonstration"""
    # Simulate head positions (roughly head-shaped)
    n_points = 100
    
    # Create a roughly spherical/ellipsoidal shape for head
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    # Add some variation to make it more realistic
    r = 1 + 0.2 * np.random.randn(n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)  
    z = r * np.cos(phi) * 0.8  # Slightly flattened
    
    # Add some noise to make it more realistic
    x += 0.1 * np.random.randn(n_points)
    y += 0.1 * np.random.randn(n_points)
    z += 0.1 * np.random.randn(n_points)
    
    # Generate some sample values (could be anything - temperature, pressure, etc.)
    values = np.sin(2*theta) * np.cos(phi) + 0.5 * np.random.randn(n_points)
    
    return x, y, z, values

# Example usage:
if __name__ == "__main__":
    # Generate sample data (replace with your actual data)
    x_data, y_data, z_data, values = generate_sample_head_data()

    # Load data from one participant
    data = loadmat('rsFC-fnirs-course/Data_for_Part_I.mat')['data']

    # Get Light Intensity, SD, and additional physiological measurements
    d = data['d'][0, 0]
    sd = data['SD'][0, 0][0,0]
    # Convert SD to a dictionary
    sd = { k: sd[k] for k in sd.dtype.names }

    x_data, y_data, z_data = sd["DetPos_3d"].T
    values = np.random.rand(len(x_data))  # Replace with actual values if available
    
    
    # Create the basic mesh visualization
    fig1, ax1 = create_3d_head_mesh(x_data, y_data, z_data, values)
    
    # Create advanced comparison
    fig2 = create_advanced_3d_mesh(x_data, y_data, z_data, values)
    
    plt.show()

# To use with your own data:
# x_data = your_x_coordinates
# y_data = your_y_coordinates  
# z_data = your_z_coordinates
# values = your_values_at_each_point
# fig, ax = create_3d_head_mesh(x_data, y_data, z_data, values)
# plt.show()