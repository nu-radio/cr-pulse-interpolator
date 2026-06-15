"""
Container to perform signal interpolation using Fourier methods.
"""

import jax

jax.config.update("jax_enable_x64", True)

from jax import tree_util
import jax.numpy as jnp
from jax_radio_tools import trace_utils, shower_utils
from dataclasses import dataclass, field
from functools import partial

from typing_extensions import Self, Union, Optional, Dict, Tuple, Any

from .interpolation_fourier import interp2d_fourier

# --------
# Module-level interpolator-evaluation wrappers
# These are jitted pure functions that accept arrays as arguments.
# They build a temporary interpolator (not stored on the object) and evaluate it.
# The `single_axis` parameter is static for performance; change if you need dynamic behavior.
# --------


@partial(jax.jit, static_argnames=("interp_type",))
def _eval_interp2d(
    pos_x: jnp.ndarray,
    pos_y: jnp.ndarray,
    data: jnp.ndarray,
    xq: jnp.ndarray,
    yq: jnp.ndarray,
    interp_type: str,
    ordered_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build a temporary interp2d_fourier interpolator and evaluate it.
    - pos_x, pos_y: shape (Nants,)
    - data: shape (Nants, ... ) depending on interp_type
    - xq, yq: query points, shape (M,)
    Returns evaluator output.
    """
    interp = interp2d_fourier(
        x=pos_x,
        y=pos_y,
        values=data,
        interp_type=interp_type,
        ordered_indices=ordered_indices,
    )
    return interp(xq, yq)


@tree_util.register_pytree_node_class
class interp2d_signal:
    """
    Interpolation module for individual traces (signals) using Fourier-based methods.

    The default units for this package are:
    - time in seconds
    - frequency in MHz

    Note that this implementation in JAX is different from the original NumPy version in the following way:
    - the timing method is not (yet?) implemented
    - only linear interpolation is available
    - since the timing method is not implemented, we do not compute for the cutoff frequency and the degree of coherency.

    Parameters
    ----------
    x : jax.typing.ArrayLike
        1D array of x positions of simulated antennas (in shower plane!)
    y : jax.typing.ArrayLike
        idem for y
    signals : jax.typing.ArrayLike
        the time traces at positions (x, y), shape (Nants, Nsamples, Npol)
    signals_start_times : jax.typing.ArrayLike or None
        the start times of the input signals at positions (x, y), shape (Nants, ), in seconds. If None, assumed to be zero for all antennas.
    sampling_period : float, default=0.1 ns
        time step between samples in seconds
    lowfreq : float, default=30 MHz
        low frequency cut for Fourier components, in MHz
    highfreq : float, default=500 MHz
        high frequency cut for Fourier components, in MHz
    lowfreq_phase : float, default=30 MHz
        low frequency cut for calculation of the phase corrections & Hilbert envelope, in MHz
    highfreq_phase : float, default=80 MHz
        high frequency cut for calculation of the phase corrections & Hilbert envelope, in MHz
    upsample_factor : int, default=5
        factor by which to upsample time traces for pulse timing search
    """

    def __init__(
        self: Self,
        signal_shape : tuple,
        ordered_indices: jax.typing.ArrayLike,
        sampling_period: float = 1.0e-10,  # in seconds
        lowfreq: float = 30.0,  # in MHz
        highfreq: float = 500.0,  # in MHz
        lowfreq_phase: float = 30.0,  # in MHz
        highfreq_phase: float = 80.0,  # in MHz,
        upsample_factor: int = 5,
    ) -> None:
        """
        Initialize all the interpolators that will be used for signal interpolation.

        This basically initializes the object with the necessary parameters,
        but doesnt actually do any computation. The computation is delegated to the `initialize` method, since we want to JIT compile that part.

        Parameters
        ----------
        signal_shape : tuple
            shape of the input signals, (Nants, Nsamples, Npol)
            NOTE: MUST BE in this shape!
        ordered_indices : jax.typing.ArrayLike
            precomputed ordering indices for the antenna positions.

            Note: this is assumed to be the same for any position set given in initialize. This may not be true in general, but is a reasonable assumption for now.

            In particular, such an assumption is required since JAX requires static arguments for JIT compilation.
        sampling_period : float, default=0.1 ns
            time step between samples in seconds
        lowfreq : float, default=30 MHz
            low frequency cut for Fourier components, in MHz
        highfreq : float, default=500 MHz
            high frequency cut for Fourier components, in MHz
        lowfreq_phase : float, default=30 MHz
            low frequency cut for calculation of the phase corrections & Hilbert envelope, in MHz
        highfreq_phase : float, default=80 MHz
            high frequency cut for calculation of the phase corrections & Hilbert envelope, in MHz
        upsample_factor : int, default=5
            factor by which to upsample time traces for pulse timing search
        """
        (Nants, Nsamples, Npols) = signal_shape  # hard assumption, 3D...
        self.trace_length = Nsamples
        self.sampling_period = sampling_period
        self.lowfreq = lowfreq
        self.highfreq = highfreq
        self.lowfreq_phase = lowfreq_phase
        self.highfreq_phase = highfreq_phase
        self.upsample_factor = upsample_factor

        self.ordered_indices = ordered_indices  # store the ordering indices in shape (Nphi, Nrad)

        # store the other shapes too
        self.Nants = Nants
        self.Npols = Npols

    def initialize(
        self: Self,
        x: jax.typing.ArrayLike,
        y: jax.typing.ArrayLike,
        signals: jax.typing.ArrayLike,
        signal_start_times: Optional[jax.typing.ArrayLike] = None,
    ) -> None:
        """
        Initialize the interpolator.

        This is separated from __init__ to allow for JIT compilation of the initialization step.

        This includes to compute the following:
        - FFT spectra of the input signals
        - Pulse timings (summed over all polarizations) using Hilbert envelope (in 30-80 MHz band)
        - Timing-corrected phase spectra
        - constant phase component in [30, 80] MHz band
        - constructing the Fourier interpolator for all quantities:
            - absolute amplitude spectrum
            - timing-corrected phase spectrum (cos and sin components)
            - pulse timings
            - constant phase component
            - arrival times, if provided

        Parameters
        ----------
        x : jax.typing.ArrayLike
            1D array of x positions of simulated antennas (in shower plane!)
        y : jax.typing.ArrayLike
            idem for y
        signals : jax.typing.ArrayLike
            the time traces at positions (x, y), shape (Nants, Nsamples, Npol)
        signals_start_times : jax.typing.ArrayLike or None
            the start times of the input signals at positions (x, y), shape (Nants, ), in seconds. If None, assumed to be zero for all antennas.
        """
        self.pos_x = jnp.asarray(x)
        self.pos_y = jnp.asarray(y)

        # first obtain the frequency spectra
        freqs, _, abs_spectrum, phase_spectrum, _ = self.get_spectra(
            signals
        )  # shapes ((Nfreq,), (Nants, Nfreq, Npol), ...)

        self.freqs = freqs  # store frequency grid

        # get the pulse timings
        pulse_timings = self.get_pulse_timings(
            signals,
            lowfreq=self.lowfreq_phase,
            highfreq=self.highfreq_phase,
            upsample_factor=self.upsample_factor,
        )  # shape (Nants, )

        phase_spectra_corrected = self.get_timing_corrected_phases(
            freqs, phase_spectrum, pulse_timings
        )  # shape (Nants, Nfreq, Npol)

        # get constant phases
        const_phases_unwrapped = self.get_constant_phases(
            abs_spectrum,
            phase_spectra_corrected,
            lowfreq=self.lowfreq_phase,
            highfreq=self.highfreq_phase,
        )  # shape (Nants, Npol)

        self.phase_spectrum_uncorrected = phase_spectrum  # (Nants, Nfreq, Npol)

        # now we calculated the corrected phase spectra, subtracted by the constant phase
        phase_spectra_corrected = (
            phase_spectra_corrected - const_phases_unwrapped[:, None, :]
        )

        # store the minimal arrays required to reconstruct signals later
        # store cos/sin of corrected phases to keep interpolation of phase stable
        eps = 1e-20  # some small number
        self.abs_spectrum = jnp.maximum(
            jnp.asarray(abs_spectrum), eps
        )  # (Nants, Nfreq, Npol)
        self.phase_cos = jnp.cos(phase_spectra_corrected)
        self.phase_sin = jnp.sin(phase_spectra_corrected)
        self.phase_spectrum = phase_spectra_corrected  # (Nants, Nfreq, Npol)

        self.pulse_timings = jnp.asarray(pulse_timings)  # (Nants,)
        self.const_phases = jnp.asarray(const_phases_unwrapped)  # (Nants, Npol)

        if signal_start_times is not None:
            self.start_times = jnp.asarray(signal_start_times)
        else:
            self.start_times = None

    # -----------------
    # PyTree helpers
    # -----------------
    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # Children (leaves) => arrays that JAX should see as dynamic
        children = (
            self.pos_x,
            self.pos_y,
            self.freqs,
            self.abs_spectrum,
            self.phase_cos,
            self.phase_sin,
            self.phase_spectrum,
            self.phase_spectrum_uncorrected,
            self.pulse_timings,
            self.const_phases,
            self.start_times,
            self.ordered_indices,
        )

        # Aux (static small metadata)
        aux = dict(
            trace_length=self.trace_length,
            sampling_period=self.sampling_period,
            lowfreq=self.lowfreq,
            highfreq=self.highfreq,
            Nants=self.Nants,
            Npols=self.Npols,
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: Dict[str, Any], children: Tuple[Any, ...]
    ) -> "interp2d_signal":
        # reconstruct object without running __init__
        obj = cls.__new__(cls)
        (
            obj.pos_x,
            obj.pos_y,
            obj.freqs,
            obj.abs_spectrum,
            obj.phase_cos,
            obj.phase_sin,
            obj.phase_spectrum,
            obj.phase_spectrum_uncorrected,
            obj.pulse_timings,
            obj.const_phases,
            obj.start_times,
            obj.ordered_indices,
        ) = children

        # restore aux
        obj.trace_length = int(aux["trace_length"])
        obj.sampling_period = float(aux["sampling_period"])
        obj.lowfreq = float(aux["lowfreq"])
        obj.highfreq = float(aux["highfreq"])
        obj.Nants = int(aux["Nants"])
        obj.Npols = int(aux["Npols"])

        return obj

    @staticmethod
    def phase_wrap(phases: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
        """
        Wrap `phases` (float or any array shape) into interval (-pi, pi).

        Parameters
        ----------
        phases : array_like
            The values to wrap into interval (-pi, pi)

        Returns
        -------
        jax.typing.ArrayLike
            The wrapped phases into interval (-pi, pi)
        """
        return (phases + jnp.pi) % (2 * jnp.pi) - jnp.pi

    @staticmethod
    def phase_unwrap_2d(
        ordered_indices : jax.typing.ArrayLike, phases: jax.typing.ArrayLike
    ) -> jax.typing.ArrayLike:
        """
        Unwrap the phases in 2D.

        Basically we unwrap along radial then azimuthal directions.

        Parameters
        ----------
        ordered_indices : jax.typing.ArrayLike
            ordering indices to reshape phases into 2D (Nradial, Nangular)
        phases : jax.typing.ArrayLike
            phases to unwrap, shape (Nants, Npol)

        Returns
        -------
        jax.typing.ArrayLike
            unwrapped phases, shape (Nants, Npol)
        """
        # indices = shower_utils.get_ordering_indices(x, y)
        ordered_phases = phases[ordered_indices]

        # First along radial axis, then angular axis
        phases_unwrapped_2d = jnp.unwrap(ordered_phases, axis=0)
        phases_unwrapped_2d = jnp.unwrap(phases_unwrapped_2d, axis=1)

        # # Back into original 1D shape...
        phases_unwrapped = jnp.zeros(phases.shape)

        (Nradial, Nangular, Npol) = phases_unwrapped_2d.shape

        phases_unwrapped = phases_unwrapped.at[ordered_indices.ravel()].set(
            phases_unwrapped_2d.reshape((Nradial * Nangular, Npol))
        )

        return phases_unwrapped

    @staticmethod
    def get_freq_mask(
        freqs: jax.typing.ArrayLike,
        low_freq: jax.typing.ArrayLike,
        high_freq: jax.typing.ArrayLike,
    ) -> jax.typing.ArrayLike:
        """
        Get a boolean mask for frequencies within [low_freq, high_freq]

        Parameters
        ----------
        freqs : jax.typing.ArrayLike
            Frequency grid in MHz
        low_freq : float
            Low frequency cut in MHz
        high_freq : float
            High frequency cut in MHz

        Returns
        -------
        jax.typing.ArrayLike
            Boolean mask for frequencies within [low_freq, high_freq]
        """
        return (freqs >= low_freq) & (freqs <= high_freq)

    def get_spectra(self: Self, signals: jnp.ndarray) -> tuple:
        """
        Do FFT of 'signals', assumed shape (Nants, Nsamples, Npol) i.e. the time traces are along the second axis.

        Produce absolute-amplitude spectrum and phase spectrum

        Parameters
        ----------
        signals : np.ndarray
            the input time traces, shaped as (Nants, Nsamples, Npol)

        Returns
        -------
        freqs : jnp.ndarray
            Frequencies corresponding to FFT components, in MHz
        all_antennas_spectrum : jnp.ndarray
            The complex FFT spectrum for all antennas, shape (Nants, Nfreq, Npol)
        abs_spectrum : jnp.ndarray
            The absolute amplitude spectrum for all antennas, shape (Nants, Nfreq, Npol)
        phasespectrum : jnp.ndarray
            The phase spectrum for all antennas, shape (Nants, Nfreq, Npol)
        unwrapped_phases : jnp.ndarray
            The unwrapped phase spectrum for all antennas, shape (Nants, Nfreq, Npol)
        """
        Nsamples = signals.shape[1]
        all_antennas_spectrum = jnp.fft.rfft(signals, axis=1)
        abs_spectrum = jnp.abs(all_antennas_spectrum)
        phasespectrum = jnp.angle(all_antennas_spectrum)
        unwrapped_phases = jnp.unwrap(phasespectrum, axis=1, discont=0.7 * jnp.pi)

        freqs = jnp.fft.rfftfreq(Nsamples, d=self.sampling_period)  # in Hz
        freqs /= 1.0e6  # in MHz

        return (
            freqs,
            all_antennas_spectrum,
            abs_spectrum,
            phasespectrum,
            unwrapped_phases,
        )

    def get_pulse_timings(
        self: Self,
        signals: jax.typing.ArrayLike,
        lowfreq: float = 30.0,
        highfreq: float = 80.0,
        upsample_factor: int = 5,
    ) -> jax.typing.ArrayLike:
        """
        Get pulse timings using a Hilbert envelope. The sum over all polarizations are taken by default.

        Note that the default bandwidths should be taken (30-80 MHz) since we want to correct the timings & phases within the region where
        the phase spectra is linear.

        Parameters
        ----------
        signals : np.ndarray
            The input time traces, shaped as (Nants, Nsamples, Npol)
        lowfreq : float, default=30 MHz
            Low frequency cut for bandpass filter in MHz
        highfreq : float, default=80 MHz
            High frequency cut for bandpass filter in MHz
        upsample_factor : int, default=5
            Upsampling factor for better timing resolution

        Returns
        -------
        pulse timings : np.ndarray
            The timing of the pulse per antenna. Shaped as (Nants, ) in units of seconds.
            Note that the timings are the same for all polarizations
        """
        # remap for jax radio tools functions
        signals_jrt = jnp.transpose(signals, (2, 0, 1))  # shape (Npol, Nants, Nsamples)

        # filter the traces
        filter_traces_jrt = trace_utils.filter_trace(
            signals_jrt,
            self.sampling_period,
            f_min=lowfreq * 1.0e6,  # in Hz
            f_max=highfreq * 1.0e6,  # in Hz
            sample_axis=2,
        )  # shape (Npol, Nants, Nsamples)

        # upsample traces
        upsampled_traces_jrt = trace_utils.resample_trace(
            filter_traces_jrt,
            dt_sample=self.sampling_period,
            dt_resample=self.sampling_period / upsample_factor,
            sample_axis=2,
        )  # shape (Npol, Nants, Nsamples * upsample_factor)

        # # shift traces to center
        shifted_traces = trace_utils.shift_trace_to_center(
            upsampled_traces_jrt, sample_axis=2
        )

        hilbert_envelope = jnp.abs(trace_utils.hilbert(shifted_traces, sample_axis=2))

        # summing over polarisations
        hilbert_sum = jnp.sqrt(jnp.sum(hilbert_envelope**2, axis=0))

        # remove unnecessary variables to save memory
        del upsampled_traces_jrt

        # finding the indicies of the pulse maximum for each antenna position
        arrival_times = (
            jnp.argmax(hilbert_sum, axis=-1) - shifted_traces.shape[2] // 2
        ) * (self.sampling_period / upsample_factor)  # shape (Nants,)

        return arrival_times

    def get_timing_corrected_phases(
        self: Self,
        freq: jax.typing.ArrayLike,
        phase_spectrum: jax.typing.ArrayLike,
        pulse_timings: jax.typing.ArrayLike,
    ) -> jax.typing.ArrayLike:
        """
        Get the timing-corrected phase spectrum.

        This basically removes the timing information from the given phase spectra.

        Parameters
        ----------
        freq : np.ndarray
            Frequency grid in MHz, shaped as (Nfreq, )
        phase_spectrum : np.ndarray
            Phase spectrum, shaped as (Nants, Nfreq, Npol)
        pulse_timings : np.ndarray
            Pulse timings per antenna in seconds, shaped as (Nants, )

        Returns
        -------
        phase_spectrum_corrected : np.ndarray
            Timing-corrected phase spectrum, shaped as (Nants, Nfreq, Npol)
        """
        phase_corrections = (
            2 * jnp.pi * (freq[None, :, None] * 1.0e6) * pulse_timings[:, None, None]
        )  # shape (Nants, Nfreq, Npol)
        phase_spectrum_corrected = (
            phase_spectrum + phase_corrections
        )  # shape (Nants, Nfreq, Npol)

        # wrap phases into [0, 2pi]
        phase_spectrum_corrected = self.phase_wrap(phase_spectrum_corrected)
        phase_spectrum_corrected = jnp.unwrap(
            phase_spectrum_corrected, axis=1, discont=0.7 * jnp.pi
        )

        return phase_spectrum_corrected

    def get_constant_phases(
        self: Self,
        abs_spectrum: jax.typing.ArrayLike,
        phase_spectrum_corrected: jax.typing.ArrayLike,
        lowfreq: float = 30.0,
        highfreq: float = 80.0,
    ) -> jax.typing.ArrayLike:
        """
        Get the Hilbert phase from the amplitude spectrum and timing-corrected phase spectrum. This corresponds to the phase constant obtained after taking out the time shift and angle from the complex spectrum.

        Parameters
        ----------
        abs_spectrum : np.ndarray
            Amplitude spectrum, shaped as (Nants, Nfreq, Npol)
        phase_spectrum_corrected : np.ndarray
            Timing-corrected phase spectrum, shaped as (Nants, Nfreq, Npol)
        lowfreq : float
            Low frequency cut for bandpass filter in MHz
        highfreq : float
            High frequency cut for bandpass filter in MHz

        Returns
        -------
        const_phases_unwrapped : np.ndarray
            The unwrapped constant phases per antenna and polarization, shaped as (Nants, Npol)
        """
        complex_spectrum = abs_spectrum * jnp.exp(
            1.0j * phase_spectrum_corrected
        )  # shape (Nants, Nfreq, Npol)
        freq_mask = self.get_freq_mask(self.freqs, lowfreq, highfreq)  # shape (Nfreq)

        const_phases = jnp.angle(
            jnp.sum(complex_spectrum * freq_mask[None, :, None], axis=1)
        )  # shape (Nants, Npol)

        const_phases_unwrapped = self.phase_unwrap_2d(
            self.ordered_indices, const_phases
        )  # shape (Nants, Npol)

        return const_phases_unwrapped

    def __call__(
        self: Self, x: jax.typing.ArrayLike, y: jax.typing.ArrayLike
    ) -> jax.typing.ArrayLike:
        """
        Compute the interpolated signals at positions (x, y).

        Different to the numpy version, this version:
        - only does linear interpolation
        - pulse centers the traces
        - returns all information (traces, timing, absolute spectra, phase spectra)

        Parameters
        ----------
        x : jax.typing.ArrayLike
            x positions to interpolate to
        y : jax.typing.ArrayLike
            y positions to interpolate to
        """
        xq = jnp.asarray(x)
        yq = jnp.asarray(y)

        # Evaluate interpolations using jitted module-level wrapper.
        abs_spectrum_q = _eval_interp2d(
            self.pos_x,
            self.pos_y,
            self.abs_spectrum,
            xq,
            yq,
            "amplitude",
            self.ordered_indices,
        )
        phase_cos_q = _eval_interp2d(
            self.pos_x,
            self.pos_y,
            self.phase_cos,
            xq,
            yq,
            "phase",
            self.ordered_indices,
        )
        phase_sin_q = _eval_interp2d(
            self.pos_x,
            self.pos_y,
            self.phase_sin,
            xq,
            yq,
            "phase",
            self.ordered_indices,
        )

        phase_spectrum = jnp.angle(phase_cos_q + 1.0j * phase_sin_q)

        pulse_timings_q = _eval_interp2d(
            self.pos_x,
            self.pos_y,
            self.pulse_timings,
            xq,
            yq,
            "fourier",
            self.ordered_indices,
        )
        const_phases_q = _eval_interp2d(
            self.pos_x,
            self.pos_y,
            self.const_phases[:, None, :],
            xq,
            yq,
            "phase",
            self.ordered_indices,
        )  # extend dimension to have (Nant, Nfreq, Npol)

        phase_spectrum -= (
            2
            * jnp.pi
            * (self.freqs[None, :, None] * 1.0e6)
            * pulse_timings_q[:, None, None]
        )
        # center the traces too
        center_pulse_dt = self.sampling_period * (self.trace_length // 2)
        phase_spectrum -= (
            2 * jnp.pi * (self.freqs[None, :, None] * 1.0e6) * center_pulse_dt
        )
        phase_spectrum += const_phases_q

        # wrap the phase spectrum
        phase_spectrum = self.phase_wrap(phase_spectrum)

        # start times
        if self.start_times is not None:
            start_times_q = _eval_interp2d(
                self.pos_x,
                self.pos_y,
                self.start_times,
                xq,
                yq,
                "fourier",
                self.ordered_indices,
            )
        else:
            start_times_q = jnp.zeros_like(pulse_timings_q)
        start_times_q = start_times_q - center_pulse_dt

        # set negative absolute values to zero
        freq_mask = (self.freqs < self.lowfreq) | (
            self.freqs > self.highfreq
        )  # True where we want to zero
        # reshape for broadcasting: (1, Nfreq, 1)
        mask_b = freq_mask[None, :, None]

        abs_spectrum_q = jnp.where(abs_spectrum_q < 1e-20, 1e-20, abs_spectrum_q)
        abs_spectrum_q = jnp.where(mask_b, 0.0, abs_spectrum_q)
        # phase_spectrum = jnp.where(mask_b, 0.0, phase_spectrum)

        # reconstruct the time traces
        spectrum = abs_spectrum_q * jnp.exp(1.0j * phase_spectrum)
        traces = jnp.fft.irfft(spectrum, n=self.trace_length, axis=1)

        return {
            "traces": traces,
            "abs_spectrum": abs_spectrum_q,
            "phase_spectrum": phase_spectrum,
            "pulse_timings": pulse_timings_q,
            "start_times": start_times_q,
        }
