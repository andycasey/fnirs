
import numpy as np
import matplotlib.pyplot as plt

def plot_seed_based_sphere_style(corr_matrix, bad_channels, clim):
    """
    Creates a 2D scatter plot of the correlation values.

    Args:
        corr_matrix (np.ndarray): Correlation matrix.
        bad_channels (list): List of channels with low SNR.
        clim (list): Color limits for the plot.
    """

    # List of Short Channels for our probe
    ss_list = [8, 29, 52, 66, 75, 92, 112, 125]

    # Add Bad channels to list of channels that will not be plotted
    if isinstance(bad_channels, np.ndarray) and bad_channels.ndim > 1:
        bad_channels = bad_channels.flatten()

    ss_list = np.unique(np.concatenate((ss_list, bad_channels)))

    # Get Values from sensorymotor ROI
    roi = 31
    corr_values = corr_matrix[roi, :]

    # Create a 2D scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(np.arange(len(corr_values)), corr_values, c=corr_values, cmap='viridis', vmin=clim[0], vmax=clim[1])
    plt.colorbar(scatter)
    ax.set_title('Seed-based Correlation')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Correlation')
    plt.show()
    return fig