import numpy as np
from load_fnirs_part_ii import load_hemodynamic_data
from spherical_projection import project_fnirs_to_sphere

# Load fNIRS data
data_path = "../rsFC-fnirs-course/Data_for_Part_II.mat"
hemo_data = load_hemodynamic_data(data_path)

# Get 3D coordinates
coords_3d = hemo_data.get_spatial_coordinates_3d()

# Let's make a movie!
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 5))

hbo = hemo_data.get_hbo_data()
hbr = hemo_data.get_hbr_data()
hbt = hemo_data.get_hbt_data()

# Original 3D positions
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

def scatter_3d(ax, coords, values, cmap='RdBu_r', s=50):
    """Helper function to create a 3D scatter plot."""
    return ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2], 
        c=values[0], 
        cmap=cmap, 
        s=s,
        vmin=np.min(values),
        vmax=np.max(values),
        edgecolor='k',
        lw=0.25
    )

scatter1 = scatter_3d(ax1, coords_3d, hbo, cmap='RdBu_r', s=50)
scatter2 = scatter_3d(ax2, coords_3d, hbr, cmap='RdBu_r', s=50)
scatter3 = scatter_3d(ax3, coords_3d, hbt, cmap='RdBu_r', s=50)

ax1.set_title('HBO Concentration')
ax2.set_title('HBR Concentration')
ax3.set_title('HBT Concentration')
for i in range(8):
    fig.tight_layout()


# Create a movie
from matplotlib.animation import FuncAnimation
def animate(t):
    """Update function for the animation."""
    scatter1.set_array(hbo[t])
    scatter2.set_array(hbr[t])
    scatter3.set_array(hbt[t])
    return scatter1, scatter2, scatter3

interval = np.diff(hemo_data.time)[0] * 1000 # real-time interval in milliseconds

preferred_movie_length = 30 # seconds
interval = preferred_movie_length * 1000 / hemo_data.time.size

ani = FuncAnimation(fig, animate, frames=hemo_data.time.size, interval=interval)
ani.save('concentration.mp4', writer='ffmpeg')

