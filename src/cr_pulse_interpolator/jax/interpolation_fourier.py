# Module for Fourier interpolation of 2D functions sampled on a polar grid
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2020 - 2023
#
# See article: A. Corstanje et al. 2023, JINST 18 P09005, arXiv 2306.13514, doi 10.1088/1748-0221/18/09/P09005
# Please cite this when using code and/or methods in your analysis
import jax

jax.config.update("jax_enable_x64", True)
from typing_extensions import Self, Any, Dict, Tuple
from functools import partial
import jax.numpy as jnp
from jax import tree_util

from .utilities import (
    batched_fourier_interp_1d,
    batched_fourier_signal_interp,
    batched_fourier_amplitude_interp,
    batched_fourier_sum,
    batched_fourier_sum_1d,
)

@partial(jax.jit, static_argnames=("interp_type", "meshgrid"))
def _eval_radial_interpolator(
    rad: jnp.ndarray,
    rad_grid: jnp.ndarray,
    fft_grid: jnp.ndarray,
    interp_type: bool,
    meshgrid: bool,
):
    """
    Module-level jitted wrapper to call the radial interpolator.
    - rad: query radii (scalar or array)
    - rad_grid: (R,)
    - fft_grid: (R, nphi, ...)  (the angular_FFT arranged per-radius)
    single_axis, meshgrid are static for stable compilation.
    """
    if interp_type == 'fourier':
        return batched_fourier_interp_1d(rad=rad, rad_grid=rad_grid, fft_grid=fft_grid)
    elif interp_type == 'phase':
        return batched_fourier_signal_interp(rad=rad, rad_grid=rad_grid, fft_grid=fft_grid)
    elif interp_type == 'amplitude':
        return batched_fourier_amplitude_interp(rad=rad, rad_grid=rad_grid, fft_grid=fft_grid)


class interp2d_fourier:
    """
    Produce a callable instance (given by the function __call__) to interpolate a function value(x, y) sampled at the input positions (x, y).

    Parameters
    ----------
    x : np.ndarray
        1D array of x positions of simulated antennas
    y : np.ndarray
        idem for y
    values : np.ndarray
        the function values (as 1D array) at positions (x, y)
    single_axis : bool, default=True
    """

    @classmethod
    def cos_sin_components(cls, fourier, axis=-1):
        """Convert complex FFT to real-valued cos, sin components.
        `axis` selects which axis holds the Fourier modes (last for 1D NumPy data,
        1 for multi-D JAX batched output)."""
        cos_components = 2 * jnp.real(fourier)
        # Build index tuples to select element 0 and -1 along `axis`
        sl_first = [slice(None)] * fourier.ndim
        sl_last  = [slice(None)] * fourier.ndim
        sl_first[axis] = 0
        sl_last[axis]  = -1
        cos_components = cos_components.at[tuple(sl_first)].set(
            cos_components[tuple(sl_first)] * 0.5
        )
        cos_components = cos_components.at[tuple(sl_last)].set(
            cos_components[tuple(sl_last)] * 0.5
        )
        sin_components = -2 * jnp.imag(fourier)
        return cos_components, sin_components

    def __init__(
        self: Self,
        x: jax.typing.ArrayLike,
        y: jax.typing.ArrayLike,
        values: jax.typing.ArrayLike,
        ordered_indices: jax.Array,
        interp_type: str = 'fourier',
        meshgrid: bool = False,
    ) -> None:
        """
        Initialize the interpolator.

        Note that this differs from the numpy version, where instead the grid for the interpolator (x and y-values) are set, and that the actual interpolation is done in __call__.
        """
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        values = jnp.asarray(values)

        # Convert (x, y) to (r, phi), make 2d position array, sorting positions and values by r and phi
        radius = jnp.sqrt(x**2 + y**2)

        # Store the (unique) radius values
        radial_axis = radius[ordered_indices][:, 0]
        # Check if the radius does not vary along angular direction (with tolerance)
        # if jnp.max(jnp.std(radius[ordered_indices], axis=1)) > 0.1 * jnp.min(radius):
        #     raise ValueError(
        #         "Radius must be (approx.) constant along angular direction. Are you sure that you are using a starshape and the core is set properly?"
        #     )
        # FFT over the angular direction, for each radius
        angular_FFT = jnp.fft.rfft(values[ordered_indices], axis=1)
        fourier_norm = float(values[ordered_indices].shape[1])
        angular_FFT /= fourier_norm  # normalize

        # store minimal arrays as leaves
        self.radial_axis = radial_axis          # shape (R,)
        self.angular_FFT = angular_FFT          # shape (R, nphi, ...)
        # store phi0 (the reference offset used in __call__)
        phi = jnp.arctan2(y, x)
        phi = jnp.around(phi, 15)
        phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
        # we need the phi0 used in ordering: take the first after sorting
        phi_sorting = jnp.argsort(phi)
        self._phi0 = phi[phi_sorting][0]

        # static flags: keep as small python values in aux via tree_flatten
        self._interp_type_flag = str(interp_type)
        self._meshgrid_flag = bool(meshgrid)

    # PyTree protocol: children (leaves) and aux (static metadata)
    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        children = (self.radial_axis, self.angular_FFT, self._phi0)
        aux = {
            "interp_type": self._interp_type_flag,
            "meshgrid": self._meshgrid_flag,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: Dict[str, Any], children: Tuple[Any, ...]) -> "interp2d_fourier":
        obj = cls.__new__(cls)
        obj.radial_axis, obj.angular_FFT, obj._phi0 = children
        obj._interp_type_flag = bool(aux["interp_type"])
        obj._meshgrid_flag = bool(aux["meshgrid"])
        return obj

    def __call__(self, x, y, max_fourier_mode=None):
        """
        Interpolate the input used in __init__ for input positions (x, y).

        Parameters
        ----------
        x : float or np.ndarray
            x positions as float or numpy ND array
        y : float or np.ndarray
            idem for y
        max_fourier_mode : int, optional
            cutoff for spatial frequencies along circles, i.e. do Fourier sum up to (incl.) this mode.
            Default None i.e. do all modes
        """
        x_q = jnp.asarray(x)
        y_q = jnp.asarray(y)

        rad_q = jnp.sqrt(x_q ** 2 + y_q ** 2)
        phi_q = jnp.arctan2(y_q, x_q) - self._phi0
        # ensure array-like phi for broadcasting
        phi_q = jnp.asarray(phi_q) if jnp.isscalar(phi_q) else phi_q

        # Interpolate Fourier components over all values of radius
        # Evaluate radial interpolator with the module-level jitted wrapper
        fourier = _eval_radial_interpolator(rad_q, self.radial_axis, self.angular_FFT, self._interp_type_flag, self._meshgrid_flag)
        # decide which axis contains Fourier components depending on meshgrid
        fourier_comp_axis = 2 if self._meshgrid_flag else 1
        fourier_len = fourier.shape[fourier_comp_axis]

        # convert to cos/sin
        cos_components, sin_components = interp2d_fourier.cos_sin_components(
    fourier, axis=fourier_comp_axis   # fourier_comp_axis = 1 for non-meshgrid
)

        # determine Fourier mode multipliers
        limit = (max_fourier_mode + 1) if (max_fourier_mode is not None) else fourier_len
        mult = jnp.linspace(0, limit-1, limit, dtype=int)
        phi_k = phi_q[..., jnp.newaxis] * mult  # shape (..., limit)

        # choose correct summation helper
        fourier_summer = batched_fourier_sum if self._meshgrid_flag else batched_fourier_sum_1d

        # compute sum_k( c_k cos(k phi) + s_k sin(k phi) )
        # Slice on the Fourier modes axis, not the last axis
        sl = [slice(None)] * cos_components.ndim
        sl[fourier_comp_axis] = slice(0, limit)
        result = fourier_summer(phi_k, cos_components[tuple(sl)], sin_components[tuple(sl)])

        return result

    # Some getters for the angular FFT, its radial interpolator function, and the radial axis points used

    def get_angular_FFT(self):
        return self.angular_FFT

    def get_radial_axis(self):
        return self.radial_axis
