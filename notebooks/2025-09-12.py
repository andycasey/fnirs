# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():

    import marimo as mo
    file_browser = mo.ui.file_browser(
        filetypes=(".snirf", )
    )
    mo.vstack([file_browser])
    return file_browser, mo


@app.cell
def _(file_browser):
    import numpy as np
    import jax.numpy as jnp
    import altair as alt
    import pandas as pd

    from time import time
    from fnirs.io import load_snirf_data
    from fnirs.model import fit
    from fnirs.spherical_projection import (
        project_fnirs_to_sphere, 
        cartesian_to_spherical,
        project_to_sphere
    )

    data = load_snirf_data(file_browser.path(index=0))

    indices = np.array([
        c.channel_idx for c in data.get_channels_by_data_type_label("HbO")
    ])

    coords_3d = data.get_spatial_coordinates_3d()[indices]

    sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
    θ, ϕ = jnp.array(sphere_result['theta']), jnp.array(sphere_result['phi'])
    Y = jnp.array(data.time_series.T[indices])
    t = jnp.array(data.time)
    return Y, alt, data, fit, indices, np, pd, t, time, θ, φ


@app.cell
def _(data):
    data.stimulus[0].data[:, 0]
    return


@app.cell
def _(alt, mo, np, pd):
    # Create 3D-style scatter plot with interactive selection
    def create_3d_scatter(data, indices, y_predicted):

        x, y, z = data.get_spatial_coordinates_3d()[indices].T
        t = np.array(data.time)
    
        # Create HbO data for each channel (random for demo)
        hbo_data = list(data.time_series[:, indices].T)
        df = pd.DataFrame({
            'x': x,
            'y': y, 
            'z': z,
            'category': indices,
            't': [t] * len(x),  # Store time array for each channel
            'HbO': hbo_data,     # Store HbO array for each channel
            'model_HbO': list(y_predicted)
        })
    
    
        # Create a multi-selection that works across all views
        # Using 'category' field since that contains your indices
        brush = alt.selection_multi(fields=['category'])

        # Base chart configuration with the selection
        base = alt.Chart(df).add_selection(brush).add_selection(
            alt.selection_interval(bind='scales')
        )
        if data.stimulus:
            sdf = pd.DataFrame({
                't_min': data.stimulus[0].data[:, 0],
                't_max': data.stimulus[0].data[:, 0] + data.stimulus[0].data[:, 1],        
            })
            stimulus_overlay = alt.Chart(sdf).mark_rect(opacity=0.6, color='lightgray').encode(
                x='t_min:Q',
                x2='t_max:Q',
                y=alt.value(0),     # Start at bottom
                y2=alt.value(300)   
            )
        else:
            stimulus_overlay = None
        
        # XY projection (looking down Z-axis)
        xy_plot = base.mark_circle().encode(
            x=alt.X('x:Q', title='X Axis', scale=alt.Scale(nice=True)),
            y=alt.Y('y:Q', title='Y Axis', scale=alt.Scale(nice=True)),
            color=alt.Color('category:N', 
                           scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                           legend=None),
            size=alt.Size('z:Q', 
                         scale=alt.Scale(range=[50, 400]),
                         legend=None),
            opacity=alt.condition(brush, alt.value(0.9), alt.value(0.3)),
            stroke=alt.condition(brush, alt.value('red'), alt.value('black')),
            strokeWidth=alt.condition(brush, alt.value(2), alt.value(0.5)),
            tooltip=['x:Q', 'y:Q', 'z:Q', 'category:N']
        ).properties(
            width=300,
            height=300,
            title="XY Projection (Z as size)"
        )

        # Create timeseries df in long format for the selected points
        # First, melt the HbO data to create a long format
        timeseries_data = []
        for i, (idx, row) in enumerate(df.iterrows()):
            category = row['category']
            hbo_values = row['HbO']  # This should be an array
            t_values = row['t'] if isinstance(row['t'], (list, np.ndarray)) else np.arange(len(hbo_values))
        
            for j, (t_val, hbo_val) in enumerate(zip(t_values, hbo_values)):
                timeseries_data.append({
                    'category': category,
                    'time': t_val,
                    'HbO': hbo_val,
                    'model_HbO': y_predicted[i, j]
                })
    
        timeseries_df = pd.DataFrame(timeseries_data)
    
        # Create the timeseries plot that responds to selection
        timeseries_base = alt.Chart(timeseries_df)
    
        timeseries_plot = (
            timeseries_base.mark_line().encode(
                x=alt.X('time:Q', title='Time'),
                y=alt.Y('HbO:Q', title='HbO Signal'),
                color=alt.Color('category:N', 
                               scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                               legend=alt.Legend(title="Channel")),
                opacity=alt.condition(brush, alt.value(0.9), alt.value(0.1))
            )
            + 
            timeseries_base.mark_line().encode(
                x=alt.X('time:Q'),
                y=alt.Y('model_HbO:Q'),
                strokeDash=alt.value([10, 3]),
                opacity=alt.condition(brush, alt.value(0.9), alt.value(0.1))            
            )
        ).transform_filter(
            brush  # This filters the timeseries to only show selected categories
        ).properties(
            width=600,
            height=300,
            title="Time Series for Selected Channels"
        ).resolve_scale(
            color='shared'
        )

    
        if stimulus_overlay is not None:
            timeseries_plot = stimulus_overlay + timeseries_plot
        # Combine the plots
        chart = (xy_plot | timeseries_plot).resolve_scale(color='shared')
        return chart




    # Example of how to use it with Marimo for interactivity
    def create_interactive_scatter_with_timeseries(data, indices, Y_predicted):
        """
        Create the complete interactive visualization with scatter plots and time series
        """
        # Create the chart with scatter plots and line plot
        chart = create_3d_scatter(data, indices, Y_predicted)

        # Use Marimo's altair chart with selection tracking
        chart_widget = mo.ui.altair_chart(chart)

        return chart_widget


    # Function to handle selection and update time series plot
    def process_selected_points(selection_data):
        """
        Process the selected points and extract channel indices
        """
        if selection_data is None or len(selection_data) == 0:
            print("No points selected")
            return []
    
        selected_categories = [point['category'] for point in selection_data]
        print(f"Selected channels: {selected_categories}")
        return selected_categories


    # Complete usage example with reactive time series:
    def create_complete_visualization(data, indices, Y_predicted):
        """
        Create a complete reactive visualization that updates time series based on selection
        """
        chart_widget = create_interactive_scatter_with_timeseries(data, indices, Y_predicted)
    
        # If you want to react to selection changes in Marimo:
        # selected_points = process_selected_points(chart_widget.value)
    
        return chart_widget

    # Usage:
    # chart_widget = create_complete_visualization(data, indices)
    # chart_widget
    return (create_complete_visualization,)


@app.cell
def _(Y_predicted, create_complete_visualization, data, indices):
    import warnings
    warnings.simplefilter(action="ignore")
    chart_widget = create_complete_visualization(data, indices, Y_predicted)
    chart_widget
    return


@app.cell
def _(Y_predicted, data):
    print(Y_predicted.shape)
    print(data.time_series.shape)

    return


@app.cell
def _(Y, fit, t, time, θ, φ):
    from sklearn.metrics import r2_score
    n_fourier_components = None
    n_spherical_harmonics = 4

    t_solve = -time()
    X, f, A, ST, terms = fit(
        t, θ, ϕ, Y,
        n_spherical_harmonics, 
        n_fourier_components or len(t) // 2 + 1
    )
    t_solve += time()
    print(f"Solve time: {t_solve:.2f} s")

    Y_predicted = f(X)
    r2 = r2_score(Y, Y_predicted, multioutput='raw_values')
    print(f"R^2: {r2.mean():.3f} +/- {r2.std():.3f}")
    return (Y_predicted,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
