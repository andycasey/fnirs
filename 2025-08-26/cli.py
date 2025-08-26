import typer
from pathlib import Path
from typing import Sequence
from typing_extensions import Annotated

app = typer.Typer()

@app.command()
def main():
    """Main command for the CLI."""
    typer.echo("This is the main command!")

@app.command()
def fit(
    input_path: Annotated[Path, typer.Argument(help="Input file path")],
    output_path: Annotated[Path, typer.Option(help="Output file path")] = None,
    sph: Annotated[int, typer.Option(help="Maximum spherical harmonic degree")] = 4,
    f: Annotated[int, typer.Option(help="Number of Fourier components")] = None,
    elevation: Annotated[float, typer.Option(help="Elevation angle for 3D view")] = 10.0,
    azimuth: Annotated[float, typer.Option(help="Azimuth angle for 3D view")] = 225.0,
):
    """Fit a model to a processed SNIRF data set."""

    from time import time
    import numpy as np
    import jax.numpy as jnp
    from spherical_projection import project_fnirs_to_sphere, cartesian_to_spherical
    from snirf import load_snirf_data
    from model import fit, create_spherical_harmonics_basis
    from sklearn.metrics import r2_score

    data = load_snirf_data(input_path)

    # TODO: Not sure why I can't do this by 
    # data.get_channels_by_data_type(), because the HbO channels
    # have data_type=99999.
    # ACTION: Check the SNIRF file + spec, and cross-check with Androu
    indices = np.array([
        c.channel_idx for c in data.get_channels_by_data_type_label("HbO")
    ])

    coords_3d = data.get_spatial_coordinates_3d()[indices]
    Y = data.time_series.T[indices]

    sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
    θ, ϕ = jnp.array(sphere_result['theta']), jnp.array(sphere_result['phi'])

    t = jnp.array(data.time)
    f = f or len(t)

    Y = jnp.array(Y)

    t_solve = -time()
    X, f, A, ST, terms = fit(t, θ, ϕ, Y, sph, f or len(t))
    t_solve += time()
    print(f"Solve time: {t_solve:.2f} s")

    Y_predicted = f(X)
    r2 = r2_score(Y, Y_predicted, multioutput='raw_values')
    print(f"R^2: {r2.mean():.3f} +/- {r2.std():.3f}")
    
    from skull_mesh_refiner import AdvancedSkullMeshRefinement


    refiner = AdvancedSkullMeshRefinement(data.get_spatial_coordinates_3d()[indices])
    #fine_points, mesh = refiner.method_pymeshlab_refinement() # not bad
    #fine_points, mesh = refiner.method_trimesh_refinement()
    fine_points, mesh = refiner.method_open3d_ball_pivoting()
    r, theta, phi = cartesian_to_spherical(fine_points, center=sphere_result["sphere_center"])

    SiT, terms = create_spherical_harmonics_basis(
        theta, 
        phi, 
        max_degree=sph
    )
    Yi = (A(X) @ SiT.T).T

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20,10 ))
    gs = gridspec.GridSpec(
        2, 
        3,
        width_ratios=[1, 1, 3], 
        height_ratios=[15, 1], 
        hspace=0.2,
        left=0.02,
        right=0.98
    )

    ax_data = fig.add_subplot(gs[0, 0], projection='3d')
    ax_model = fig.add_subplot(gs[0, 1], projection='3d')
    
    ax_channels = fig.add_subplot(gs[:1, 2])
    ax_cbar = fig.add_subplot(gs[1, :2])  # Colorbar spans first two columns

    ax_data.set_title("Data")
    ax_model.set_title("Model")
    for spine in ("right", "top", "left"):
        ax_channels.spines[spine].set_visible(False)
    ax_channels.set_xlabel("Time [s]")
    ax_channels.set_yticks([])
    

    for ax in (ax_data, ax_model):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Set view to show right side of brain (azimuth=0 looks from positive y-axis)
        ax.view_init(elev=elevation, azim=azimuth)


    def scatter_3d(ax, coords_3d, values, **kwargs):
        """Helper function to create a 3D scatter plot."""
        return ax.scatter(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], 
            c=values, 
            **kwargs
        )
    
    # Plot all the channels
    for i in range(Y.shape[0]):
        bias = np.mean(Y[i])
        scale = np.ptp(Y[i])
        ax_channels.plot(t, (Y[i] - bias)/scale + i, c='k', alpha=0.5)
        ax_channels.plot(t, (Y_predicted[i] - bias)/scale + i, c='tab:green', alpha=0.5)
    
    # Add vertical lines to indicate current time
    t_line = ax_channels.axvline(t[0], c="#666666", ls=":")


    scatter_kwargs = dict(
        cmap='RdBu_r', 
        s=30,
        vmin=np.percentile(Y, 5),
        vmax=np.percentile(Y, 95),
        edgecolor='k',
        lw=0.25
    )
    scatter_data = scatter_3d(ax_data, coords_3d, Y[:, 0], **scatter_kwargs)
    scatter_model = scatter_3d(ax_model, fine_points, Yi[:, 0], **scatter_kwargs)

    # Apply tight_layout before adding colorbar
    fig.tight_layout()
    
    # Add colorbar in the dedicated space
    cbar = plt.colorbar(
        scatter_data, 
        cax=ax_cbar,
        orientation='horizontal',
        aspect=40,
        shrink=0.8
    )
    cbar.set_label(f"HbO concentration change")
        
    from tqdm import tqdm
    with tqdm(total=data.time.size, desc="Creating animation") as pb:
        def animate(t):
            """Update function for the animation."""
            scatter_data.set_array(Y[:, t])
            scatter_model.set_array(Yi[:, t])
            t_line.set_xdata([data.time[t], data.time[t]])
            pb.update()
            return (scatter_data, scatter_model, t_line)
        


        preferred_movie_length = 30 # seconds
        interval = preferred_movie_length * 1000 / data.time.size
        fig.tight_layout()
        # Keep a reference to the animation to prevent garbage collection
        ani = FuncAnimation(fig, animate, frames=data.time.size, interval=interval, blit=False, repeat=True)
        if output_path is not None:
            ani.save(output_path, writer='ffmpeg', dpi=150)
        else:
            fig._animation = ani
            plt.show()

    # Store animation in the figure to keep it alive
    #fig._animation = ani
    
    #plt.show()

'''
@app.command()
def fit_general(
    input_path: Annotated[Path, typer.Argument(help="Input file path")],
    n_physiological_eigenvectors: Annotated[int, typer.Option(help="Number of physiological eigenvectors to remove")] = 2,
    physiological_linear_model: Annotated[bool, typer.Option(help="Whether to use a linear model to regress out physiological indicators")] = True,
    spherical_degree: Annotated[int, typer.Option(help="Maximum spherical harmonic degree")] = 4,
    fourier_components: Annotated[int, typer.Option(help="Number of Fourier components")] = None
):
    """Fit command for the CLI."""



    import numpy as np
    import jax.numpy as jnp
    from time import time
    from physiological import regress_physiological_signals
    from spherical_projection import project_fnirs_to_sphere
    from model import fit

    if f"{input_path}".lower().endswith(".snirf"):
        from snirf import load_snirf_data
        data = load_snirf_data(input_path)
        # TODO: not sure how to handle unprocessed snirf data yet

        # TODO: Not sure why I can't do this by 
        # data.get_channels_by_data_type(), because the HbO channels
        # have data_type=99999.
        # ACTION: Check the SNIRF file + spec, and cross-check with Androu
        indices = np.array([
            c.channel_idx for c in data.get_channels_by_data_type_label("HbO")
        ])

        coords_3d = data.get_spatial_coordinates_3d()[indices]
        Y = data.time_series.T[indices]

    else:
        from load_fnirs_part_ii import load_hemodynamic_data
        data = load_hemodynamic_data(input_path)

        # TODO: just get the short channels from the distances in the data file
        short_channel_indices = np.array([7, 28, 51, 65, 74, 91, 111, 124])


        is_short_channel = np.zeros(len(data.channels), dtype=bool)
        is_short_channel[short_channel_indices] = True

        if n_physiological_eigenvectors > 0 or physiological_linear_model:

            # Load data from input_path
            Y = regress_physiological_signals(
                data,
                short_channel_indices=short_channel_indices,
                remove_n_eigenvectors=n_physiological_eigenvectors
            ) 
        else:
            Y = data.get_hbo_data()[:, ~is_short_channel]
        coords_3d = data.get_spatial_coordinates_3d()[~is_short_channel]




    sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
    θ, ϕ = jnp.array(sphere_result['theta']), jnp.array(sphere_result['phi'])

    t = jnp.array(data.time)
    fourier_components = fourier_components or len(t)
    fourier_components = 64

    Y = jnp.array(Y)

    t_solve = -time()
    X, f, A, ST, terms = fit(
        t, θ, ϕ, Y, 
        max_spherical_degree=spherical_degree, 
        n_fourier_components=fourier_components
    )
    t_solve += time()
    print(f"Solve time: {t_solve:.2f} s")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(θ, ϕ, s=30)


    plt.show()

    Y_predicted = f(X)

    scales = []
    biases = []
    fig, axes = plt.subplots(3)
    for i in range(Y.shape[0]):
        bias = np.mean(Y[i])
        scale = np.ptp(Y[i])
        axes[0].plot(t, (Y[i] - bias)/scale + i, c='k')
        scales.append(scale)
        biases.append(bias)
    
    axes[1].hist(scales, bins=30)
    axes[2].hist(biases, bins=30)
    

    indices = np.random.choice(Y.shape[0], size=10, replace=False)



    fig, ax = plt.subplots()
    for i, index in enumerate(indices):
        bias = np.mean(Y[index])
        scale = np.ptp(Y[index])
        ax.plot(t, (Y[index] - bias)/scale + i, c='k')
        ax.plot(t, (Y_predicted[index] - bias)/scale + i, c='tab:red')
        ax.text(t[-1], i, f"Channel {index + 1}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(Y, Y_predicted, s=1, alpha=0.1)
    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.min(limits), np.max(limits)
    ax.plot(limits, limits, color='k', ls='--')
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    abs_residuals = np.abs(Y - Y_predicted)
    scale = 1

    # Plot sum of absolute residuals on the sphere
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121)
    ax.plot(t, abs_residuals.sum(axis=0) / abs_residuals.shape[1], c='k')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Mean of absolute residuals per channel ({1/scale:.1e})')


    ax = fig.add_subplot(122, projection='3d')
    scat = ax.scatter(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        c=abs_residuals.sum(axis=1) / abs_residuals.shape[1],
        cmap='viridis', 
        s=50
    )
    plt.colorbar(scat, ax=ax, label=f'Mean of absolute residuals per second ({1/scale:.1e})')
    for i in range(4):
        fig.tight_layout()

    plt.show()
    raise a
'''



if __name__ == "__main__":
    app()

