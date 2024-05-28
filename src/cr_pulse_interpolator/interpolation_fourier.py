# Module for Fourier interpolation of 2D functions sampled on a polar grid
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2020 - 2023
#
# See article: A. Corstanje et al. 2023, JINST 18 P09005, arXiv 2306.13514, doi 10.1088/1748-0221/18/09/P09005 
# Please cite this when using code and/or methods in your analysis

import numpy as np
import scipy.interpolate as intp

class interp2d_fourier:
    @classmethod
    def get_ordering_indices(self, x, y):
        """ Produces ordering indices to create (radius, phi) 2D-array from unordered x and y (1D-)arrays.

        Parameters
        ----------
        x : 1D array of x positions
        y : 1D array of y positions
        """

        radius = np.sqrt(x**2 + y**2)

        phi = np.arctan2(y, x)  # uses interval -pi..pi
        phi = np.around(phi, 15)  # based on observation that offsets from 0 up to 1e-16 can result from arctan2
        phi[phi<0] += 2*np.pi  # put into 0..2pi for ordering.

        phi_sorting = np.argsort(phi)
        # Assume star-shaped pattern, i.e. radial # steps = number of (almost) identical phi-values
        # May not work very near (0, 0)
        self._phi0 = phi[phi_sorting][0]
        #print('Phi0 = %1.8f' % self._phi0)
        test = phi[phi_sorting] - self._phi0
        radial_steps = len(np.where(np.abs(test) < 0.0001)[0])
        phi_steps = len(phi_sorting) // radial_steps
        phi_sorting = phi_sorting.reshape((phi_steps, radial_steps))
        indices = np.argsort(radius[phi_sorting], axis=1)
        for i in range(phi_steps): # Sort by radius; should be possible without for-loop...
            phi_sorting[i] = phi_sorting[i][indices[i]]
        ordering_indices = phi_sorting.T # get shape (radial_steps, phi_steps)

        return ordering_indices

    @classmethod
    def cos_sin_components(cls, fourier):
        """
        Convert complex FFT as from np.fft.rfft to real-valued cos, sin components
        Input: complex Fourier components, with Fourier series running along the last axis.
        """
        cos_components = 2*np.real(fourier)
        cos_components[..., 0] *= 0.5
        cos_components[..., -1] *= 0.5
        sin_components = -2*np.imag(fourier)

        return (cos_components, sin_components)

    def __init__(self, x, y, values, radial_method='cubic', fill_value=None, recover_concentric_rings=False):
        """
        # Produce a callable instance (given by the function __call__) to interpolate a function value(x, y) sampled at the input positions (x, y)

        Parameters
        ----------
        x : 1D array of x positions of simulated antennas
        y : idem for y
        values : the function values (as 1D array) at positions (x, y)
        radial_method : the interp1d method for interpolating Fourier components along the radial axis (optional, default 'cubic' i.e. cubic splines)
        fill_value : the fill value to pass to interp1d to use for a radius outside the min..max radius interval from the input. Set to 'extrapolate' to extrapolate beyond radial limits; accuracy outside the interval is limited. Optional, default None, meaning stay constant for r < r_min, and 0 for r > r_max
        recover_concentric_rings : set True if the grid is not purely circular-symmetric; results may not be accurate. Optional, default False
        """
        # Convert (x, y) to (r, phi), make 2d position array, sorting positions and values by r and phi
        radius = np.sqrt(x**2 + y**2)

        ordering_indices = self.get_ordering_indices(x, y)
        values_ordered = np.copy(values)[ordering_indices]

        # Store the (unique) radius values
        self.radial_axis = radius[ordering_indices][:, 0]
        # Check if the radius does not vary along angular direction (with tolerance)
        if np.max(np.std(radius[ordering_indices], axis=1)) > 0.1 * np.min(radius):
            if not recover_concentric_rings:
                raise ValueError("Radius must be (approx.) constant along angular direction." + \
                                " You can try to \"fix\" that by using \"recover_concentric_rings=True\"")
            else:
                self.radial_axis = np.mean(radius[ordering_indices], axis=1)
                values_ordered_interpolated = []
                for x, y in zip(radius[ordering_indices].T, values_ordered.T):
                    intpf = intp.interp1d(
                        x, y, axis=0, kind=radial_method, fill_value='extrapolate')
                    values_ordered_interpolated.append(intpf(self.radial_axis))
                values_ordered = np.array(values_ordered_interpolated).T

        # FFT over the angular direction, for each radius
        self.angular_FFT = np.fft.rfft(values_ordered, axis=1)
        length = values_ordered.shape[-1]
        self.angular_FFT /= float(length) # normalize

        # Produce interpolator function, interpolating the FFT components as a function of radius
        self.interpolator_radius = intp.interp1d(self.radial_axis, self.angular_FFT, axis=0, kind=radial_method, fill_value='extrapolate') # Interpolates the Fourier components along the radial axis


    def __call__(self, x, y, max_fourier_mode=None):
        """
        Interpolate the input used in __init__ for input positions (x, y)

        Parameters
        ----------
        x : x positions as float or numpy ND array
        y : idem for y
        max_fourier_mode : optional cutoff for spatial frequencies along circles, i.e. do Fourier sum up to (incl.) this mode. Default None i.e. do all modes
        """
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) - self._phi0

        # Interpolate Fourier components over all values of radius
        fourier = self.interpolator_radius(radius)
        fourier_len = fourier.shape[-1]

        (cos_components, sin_components) = interp2d_fourier.cos_sin_components(fourier)

        limit = max_fourier_mode+1 if max_fourier_mode is not None else fourier_len
        mult = np.linspace(0, limit-1, limit).astype(int) # multipliers for Fourier modes, as k in cos(k*phi), sin(k*phi)
        result = np.zeros_like(radius)
        if isinstance(phi, float):
            result = np.sum(cos_components[..., 0:limit] * np.cos(phi * mult)) + np.sum(sin_components[..., 0:limit] * np.sin(phi * mult))
        else:
            result = np.sum(cos_components[..., 0:limit] * np.cos(phi[..., np.newaxis] * mult), axis=-1) + np.sum(sin_components[..., 0:limit] * np.sin(phi[..., np.newaxis] * mult), axis=-1)
        # The Fourier sum done explicitly, as sum_k( c_k cos(k phi) + s_k sin(k phi) )

        return result

    # Some getters for the angular FFT, its radial interpolator function, and the radial axis points used

    def get_angular_FFT(self):

        return self.angular_FFT

    def get_angular_FFT_interpolator(self):

        return self.interpolator_radius

    def get_radial_axis(self):

        return self.radial_axis
