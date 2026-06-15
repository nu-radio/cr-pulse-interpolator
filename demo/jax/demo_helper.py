# Separate module for cross-correlating and making a demo plot, JAX version
# Adapted from demo_helper.py for use with cr_pulse_interpolator.jax

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from jax_radio_tools.shower_utils import get_ordering_indices as _get_ordering_indices


def get_freq_axis(signal, sampling_period=0.1e-9):
    """
    Return the frequency axis of a (real) FFT spectrum of time series 'signal' as 1D array.
    Frequencies in MHz.

    Parameters
    ----------
    signal : array-like
        time series, 1D array
    sampling_period : float, default=0.1 ns
        time between samples in seconds
    """
    return 1.0e-6 * np.fft.rfftfreq(np.asarray(signal).shape[0], d=sampling_period)  # MHz


def do_filter_signal_lowpass(signal, cutoff_freq, sampling_period=0.1e-9):
    """
    For one signal time series, do lowpass filtering.

    Parameters
    ----------
    signal : array-like
        time series, 1D array
    cutoff_freq : float
        high frequency cutoff in MHz
    sampling_period : float, default=0.1 ns
        time between samples in seconds
    """
    signal = np.asarray(signal)
    freqs = get_freq_axis(signal, sampling_period)

    filter_indices = np.where(freqs > cutoff_freq)
    spectrum = np.fft.rfft(signal)
    spectrum[filter_indices] *= 0.0
    signal_filtered = np.fft.irfft(spectrum)

    return signal_filtered


def get_crosscorrelation(test_signal, orig_signal, upsampling_factor=10):
    """
    Get normalized cross-correlation between 'test_signal' and 'orig_signal', returned as 'CC_zeroshift'.
    Also returns: normalized cross-correlation optimized over time shift between the two signals;
    time difference for which the cross-correlation is maximal;
    relative energy difference.

    Parameters
    ----------
    test_signal : array-like
        time series, 1D array
    orig_signal : array-like
        time series, 1D array
    upsampling_factor : int, default=10
        upsampling factor for sub-sample timing accuracy
    """
    test_signal = np.asarray(test_signal)
    orig_signal = np.asarray(orig_signal)

    orig_signal_upsampled = scipy_signal.resample(orig_signal, upsampling_factor * len(orig_signal))
    test_signal_upsampled = scipy_signal.resample(test_signal, upsampling_factor * len(test_signal))

    crosscorr = scipy_signal.correlate(test_signal_upsampled, orig_signal_upsampled)
    normalization = np.sqrt(np.sum(orig_signal_upsampled**2) * np.sum(test_signal_upsampled**2))
    crosscorr /= normalization

    autocorr = scipy_signal.correlate(orig_signal_upsampled, orig_signal_upsampled)
    max_autocorr = np.argmax(autocorr)

    CC_optimized_timeshift = np.max(crosscorr)
    CC_zeroshift = crosscorr[max_autocorr]

    delta_t = 0.1 * (1.0 / upsampling_factor) * (np.argmax(crosscorr) - np.argmax(autocorr))  # in ns

    orig_energy = np.sum(orig_signal_upsampled**2)
    test_energy = np.sum(test_signal_upsampled**2)
    energy_rel_diff = (test_energy - orig_energy) / orig_energy

    return (CC_zeroshift, CC_optimized_timeshift, delta_t, energy_rel_diff)


def plot_pulse_and_spectrum(orig_time_axis, orig_pulse, interpolated_time_axis, interpolated_pulse, x, y, pol, sampling_period=0.1e-9, window_samples=250, fig_path=None):
    """
    Plots an interpolated pulse together with a 'true' simulated pulse.

    Note: unlike the NumPy version this function has no cutoff_freq parameter, since the JAX
    version of the signal interpolator does not compute it.

    The plot window is automatically centred on the pulse maximum of the original trace.

    Parameters
    ----------
    orig_time_axis : np.ndarray
        Time axis for the original pulse
    orig_pulse : np.ndarray
        time trace, 1D array
    interpolated_time_axis : np.ndarray
        Time axis for the interpolated pulse
    interpolated_pulse : np.ndarray
        interpolated time trace, 1D array
    x : float
        x position in m (for annotation)
    y : float
        y position in m (for annotation)
    pol : int
        polarization number (for annotation)
    sampling_period : float, default=0.1 ns
        time between samples in seconds
    window_samples : int, default=250
        half-width of the time-domain plot window in samples, centred on the pulse peak
    fig_path : str or None, default=None
        If not None, path to save the figure; if None, the figure is shown interactively
    """
    orig_pulse = np.asarray(orig_pulse)
    interpolated_pulse = np.asarray(interpolated_pulse)

    radius = np.sqrt(x**2 + y**2)
    freqs = get_freq_axis(orig_pulse, sampling_period)

    (CC_zeroshift, CC_optimized_timeshift, delta_t, energy_rel_diff) = get_crosscorrelation(orig_pulse, interpolated_pulse)

    fig, ax = plt.subplots(figsize=(10.67, 4), nrows=1, ncols=2)
    ax1, ax2 = ax[0], ax[1]

    ax1.plot(orig_time_axis, 1.0e6 * orig_pulse, label='orig pulse', lw=2)
    ax1.plot(interpolated_time_axis, 1.0e6 * interpolated_pulse, label='interpolated pulse', lw=2)

    time_offset = int(orig_pulse.argmax() - interpolated_pulse.argmax())
    residual = 1.0e6 * (np.roll(interpolated_pulse, time_offset) - orig_pulse)
    ax1.plot(orig_time_axis, residual, label='difference', lw=2, c='g')

    ax1.grid()
    ax1.set_xlabel('Time [ ns ]')
    ax1.set_ylabel(r'E-field [ $\mu$V/m ]')
    pulse_idx = int(np.argmax(np.abs(orig_pulse)))
    lo = max(0, pulse_idx - window_samples)
    hi = min(len(orig_time_axis) - 1, pulse_idx + window_samples)
    ax1.set_xlim(orig_time_axis[lo], orig_time_axis[hi])
    ax1.legend(loc='best')

    orig_pulse_powerspec = np.abs(np.fft.rfft(orig_pulse))**2
    interp_pulse_powerspec = np.abs(np.fft.rfft(interpolated_pulse))**2
    ax2.plot(freqs, orig_pulse_powerspec, label='Orig pulse')
    ax2.plot(freqs, interp_pulse_powerspec, label='Interpolated pulse')
    ax2.text(0.98, 0.40, 'Position x = %3.1f, y = %3.1f, r = %3.2f m, pol = %d' % (x, y, radius, pol),
             transform=ax2.transAxes, ha='right')
    ax2.text(0.98, 0.30, 'CC = %1.5f, CC_max = %1.5f' % (CC_zeroshift, CC_optimized_timeshift),
             transform=ax2.transAxes, ha='right')
    ax2.text(0.98, 0.20, 'delta_t = %1.2f ns' % delta_t,
             transform=ax2.transAxes, ha='right')

    ax2.grid()
    ax2.set_xlabel('Frequency [ MHz ]')
    ax2.set_ylabel('Power spectrum [ a.u. ]')
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0.0, 1.2 * np.max(orig_pulse_powerspec))
    ax2.legend(loc='best')

    plt.show()
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path,'interpolation_demo_x%3.1f_y%3.1f_pol%d.png' % (x, y, pol)), dpi=300)


def get_ordered_indices(x, y):
    """
    Compute the ordering indices for antenna positions arranged in a star-shaped pattern.

    This is required when constructing the JAX interpolator classes (interp2d_fourier and
    interp2d_signal), which take ordered_indices as a constructor argument rather than
    computing it internally.

    The returned array maps the flat list of antenna positions into a 2D grid of shape
    (Nradial, Nangular), where antennas at the same radius are grouped together.

    Parameters
    ----------
    x : array-like
        1D array of antenna x positions in m
    y : array-like
        1D array of antenna y positions in m

    Returns
    -------
    ordered_indices : np.ndarray
        2D integer array of shape (Nradial, Nangular) indexing into x and y
    """
    return _get_ordering_indices(np.asarray(x), np.asarray(y))


def read_data_hdf5(filename):
    """
    Reading in the demo data hdf5 file.
    The time traces inside are CoREAS E-fields, converted to 2 'on-sky' polarizations.
    """
    try:
        demo_file = h5py.File(filename, 'r')
    except Exception:
        raise ValueError('Cannot read data file; demo data downloaded with download_demo_data.sh?')

    zenith = demo_file.get('zenith')[()]
    azimuth = demo_file.get('azimuth')[()]
    xmax = demo_file.get('xmax')[()]
    footprint_positions = np.array(demo_file.get('footprint_positions'))
    test_positions = np.array(demo_file.get('test_positions'))
    (footprint_pos_x, footprint_pos_y) = (footprint_positions[:, 0], footprint_positions[:, 1])
    (test_pos_x, test_pos_y) = (test_positions[:, 0], test_positions[:, 1])

    footprint_antenna_data = np.array(demo_file.get('footprint_antennas'))
    test_antenna_data = np.array(demo_file.get('test_antennas'))

    footprint_time_axis = np.array(demo_file.get('time_axis_footprint_antennas'))
    test_time_axis = np.array(demo_file.get('time_axis_test_antennas'))

    demo_file.close()

    return (zenith, azimuth, xmax, footprint_pos_x, footprint_pos_y, test_pos_x, test_pos_y,
            footprint_antenna_data, test_antenna_data, footprint_time_axis, test_time_axis)
