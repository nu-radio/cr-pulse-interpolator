# Demonstration script for JAX version of interpolation_fourier.py
# Adapted from demo/demo_interpolation_fourier.py for the JAX API
# Author: A. Corstanje, (a.corstanje@astro.ru.nl), 2023

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

import cr_pulse_interpolator.jax.interpolation_fourier as interpF
from cr_pulse_interpolator.jax.utilities import batched_fourier_interp_1d

import demo_helper


def do_plot_radial(interp_fourier, max_mode=2, fig_path=None):
    """
    Plot the radial dependence of the lowest angular Fourier modes.

    For each mode k, shows the discrete values at the grid radii (dots) and the
    radially interpolated curve (lines), analogous to the numpy version.

    Parameters
    ----------
    interp_fourier : interp2d_fourier
        Initialized JAX Fourier interpolator
    max_mode : int, default=2
        Highest angular Fourier mode to plot
    fig_path : str or None, default=None
        If not None, path to save the figure; if None, the figure is shown interactively
    """
    fourier = interp_fourier.get_angular_FFT()      # (Nradial, Nmodes) complex
    radial_axis = interp_fourier.get_radial_axis()  # (Nradial,)

    fine_radius = jnp.arange(0.0, float(radial_axis[-1]) + 0.5, 0.5)
    fourier_interpolated = batched_fourier_interp_1d(
        rad=fine_radius, rad_grid=radial_axis, fft_grid=fourier
    )  # (Nfine, Nmodes) complex

    (cos_components, sin_components) = interpF.interp2d_fourier.cos_sin_components(fourier, axis=-1)
    (cos_fine, sin_fine) = interpF.interp2d_fourier.cos_sin_components(fourier_interpolated, axis=-1)

    # Convert to numpy for plotting
    radial_axis = np.asarray(radial_axis)
    fine_radius = np.asarray(fine_radius)
    cos_components = np.asarray(cos_components)
    sin_components = np.asarray(sin_components)
    cos_fine = np.asarray(cos_fine)
    sin_fine = np.asarray(sin_fine)

    plt.figure()
    for k in range(max_mode + 1):
        label = 'cos({0} phi) mode'.format(k) if k > 0 else 'Zero mode'
        plt.plot(radial_axis, cos_components[:, k], 'o', label=label)
    for k in range(max_mode + 1):
        plt.plot(fine_radius, cos_fine[:, k])
    plt.legend(loc='best')
    plt.xlabel('Radial distance [ m ]')
    plt.ylabel('Value')
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, 'radial_dependence.png'), dpi=300, bbox_inches='tight')

    plt.figure()
    for k in range(1, max_mode + 1):
        plt.plot(radial_axis, sin_components[:, k], 'o', label='sin({0} phi) mode'.format(k))
    for k in range(1, max_mode + 1):
        plt.plot(fine_radius, sin_fine[:, k])
    plt.legend(loc='best')
    plt.xlabel('Radial distance [ m ]')
    plt.ylabel('Value')
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, 'radial_dependence_sin.png'), dpi=300, bbox_inches='tight')


def do_plot_angular(interp_fourier, fixed_radius, values_for_radius, fig_path=None):
    """
    Plot angular interpolation at a fixed radius.

    Parameters
    ----------
    interp_fourier : interp2d_fourier
        Initialized JAX Fourier interpolator (meshgrid=False)
    fixed_radius : float
        The radius at which to evaluate the angular interpolation
    values_for_radius : array-like
        1D array of the original data values at this radius
    """
    phi_steps = len(values_for_radius)
    phi_step_degrees = 360.0 / phi_steps
    raw_phi_degrees = np.linspace(0.0, 360.0 - phi_step_degrees, phi_steps)

    fine_phi = np.linspace(0.0, 2 * np.pi, 1000)
    fine_points_x = fixed_radius * np.cos(fine_phi)
    fine_points_y = fixed_radius * np.sin(fine_phi)

    interp_values = np.asarray(interp_fourier(fine_points_x, fine_points_y))
    interp_values_truncated = np.asarray(
        interp_fourier(fine_points_x, fine_points_y, max_fourier_mode=2)
    )

    plt.figure()
    plt.plot(raw_phi_degrees, values_for_radius, 'o',
             label='Values at r={0:.1f} m'.format(fixed_radius))
    plt.plot(fine_phi * 180 / np.pi, interp_values, label='Fourier series')
    plt.plot(fine_phi * 180 / np.pi, interp_values_truncated, '--',
             label='Up to 2nd Fourier mode')
    plt.xlabel('Phi [ deg ]')
    plt.ylabel('Value')
    plt.legend(loc='best')
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, 'angular_dependence_r%3.1f.png' % fixed_radius), dpi=300, bbox_inches='tight')


# Load sample data — same file as the numpy version
fname = os.path.join(os.path.dirname(__file__), '..', 'sample_data.txt')
fig_path = os.path.join(os.path.dirname(__file__), 'demo_images_fourier')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
data = np.loadtxt(fname)
(x, y, values) = data.T

# Compute ordering indices required by the JAX interpolator
ordered_indices = demo_helper.get_ordered_indices(x, y)

### Create JAX interpolator for point-wise evaluation (meshgrid=False, the default)
jax_fourier_interpolator = interpF.interp2d_fourier(
    x=x, y=y, values=values, ordered_indices=ordered_indices
)
###

# Plot radial dependence of the lowest Fourier modes
do_plot_radial(jax_fourier_interpolator, max_mode=2, fig_path=fig_path)

# Plot angular interpolation at two fixed radii
radius_values = np.asarray(jax_fourier_interpolator.get_radial_axis())
for radius_stepnr in [4, 7]:
    fixed_radius = float(radius_values[radius_stepnr])
    values_for_radius = values[np.asarray(ordered_indices)][radius_stepnr, :]
    do_plot_angular(jax_fourier_interpolator, fixed_radius, values_for_radius, fig_path)

# Make color plot of f(x, y) on a meshgrid.
# The JAX interpolator requires a separate instance with meshgrid=True for
# efficient 2D evaluation (uses batched_fourier_sum instead of batched_fourier_sum_1d).
dist_scale = 250.0
ti = np.linspace(-dist_scale, dist_scale, 1000)
XI, YI = np.meshgrid(ti, ti)

### Create JAX interpolator for meshgrid evaluation
jax_fourier_interpolator_mesh = interpF.interp2d_fourier(
    x=x, y=y, values=values, ordered_indices=ordered_indices, meshgrid=True
)
ZI = np.asarray(jax_fourier_interpolator_mesh(XI, YI))
###

maxp = np.max(ZI)
fig, ax = plt.subplots()
ax.pcolor(XI, YI, ZI, vmax=maxp, vmin=0, cmap=cm.jet)
ax.scatter(x, y, marker='+', s=3, color='w')

mm = cm.ScalarMappable(cmap=cm.jet)
mm.set_array([0.0, maxp])

cbar = fig.colorbar(mm, ax=ax)
cbar.set_label('Values of f(x, y)')

ax.set_xlabel('x [ m ]')
ax.set_ylabel('y [ m ]')
ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
ax.set_aspect('equal')

plt.savefig(os.path.join(fig_path, 'interpolated_footprint.png'), bbox_inches='tight')
plt.show()
