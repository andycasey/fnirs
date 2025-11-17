
import numpy as np
import matplotlib.pyplot as plt
from .remove_autocorrelation_dc import remove_autocorrelation_dc
from .compute_correlation_coefficient import compute_correlation_coefficient
from .plot_seed_based_sphere_style import plot_seed_based_sphere_style

def plot_correlation_matrices(dc, dc_only_sc, dc_only_phys, dc_sc_phys, dc_pca_one, sd, bad_channels):
    """
    Plots correlation matrices of each preprocessing type.

    Args:
        dc (np.ndarray): Raw hemoglobin concentration changes.
        dc_only_sc (np.ndarray): Hemoglobin concentration changes after short-channel regression.
        dc_only_phys (np.ndarray): Hemoglobin concentration changes after physiological regression.
        dc_sc_phys (np.ndarray): Hemoglobin concentration changes after short-channel and physiological regression.
        dc_pca_one (np.ndarray): Hemoglobin concentration changes after PCA regression.
        sd (dict): SD structure common in fNIRS files.
        bad_channels (list): List of channels with low SNR.
    """

    # Raw case
    if dc is not None:
        pw_dc_raw = remove_autocorrelation_dc(dc, sd)
        c_raw = compute_correlation_coefficient(pw_dc_raw, bad_channels)
        plot_correlation_matrix(c_raw, 'Raw', bad_channels)

    # SC Only Case
    if dc_only_sc is not None:
        pw_dc_only_sc = remove_autocorrelation_dc(dc_only_sc, sd)
        c_only_sc = compute_correlation_coefficient(pw_dc_only_sc, bad_channels)
        plot_correlation_matrix(c_only_sc, 'Only SC', bad_channels)

    # Phys Only Case
    if dc_only_phys is not None:
        pw_dc_only_phys = remove_autocorrelation_dc(dc_only_phys, sd)
        c_only_phys = compute_correlation_coefficient(pw_dc_only_phys, bad_channels)
        plot_correlation_matrix(c_only_phys, 'Only Phys', bad_channels)

    # SC + Phys Case
    if dc_sc_phys is not None:
        pw_dc_sc_phys = remove_autocorrelation_dc(dc_sc_phys, sd)
        c_sc_phys = compute_correlation_coefficient(pw_dc_sc_phys, bad_channels)
        plot_correlation_matrix(c_sc_phys, 'SC+Phys', bad_channels)

    # PCA 1 Case
    if dc_pca_one is not None:
        pw_dc_pca_one = remove_autocorrelation_dc(dc_pca_one, sd)
        c_pca_one = compute_correlation_coefficient(pw_dc_pca_one, bad_channels)
        plot_correlation_matrix(c_pca_one, 'PCA 1', bad_channels)

def plot_correlation_matrix(c_matrix, title_prefix, bad_channels):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{title_prefix} Correlation Matrices')

    im = axes[0].imshow(c_matrix[:, :, 0], cmap='jet', vmin=-1, vmax=1)
    axes[0].set_title('HbO')
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(c_matrix[:, :, 1], cmap='jet', vmin=-1, vmax=1)
    axes[1].set_title('HbR')
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(c_matrix[:, :, 2], cmap='jet', vmin=-1, vmax=1)
    axes[2].set_title('HbT')
    fig.colorbar(im, ax=axes[2])

    plt.show()

    #fig2 = plot_seed_based_sphere_style(c_matrix[:, :, 2], bad_channels, [-.8, .8])
    #return (fig, fig2)
    return fig
