# Separate module for cross-correlating and making a demo plot
# Author: A. Corstanje, (a.corstanje@astro.ru.nl), 2023

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

def get_freq_axis(signal):
    """
    Return the frequency axis of a (real) FFT spectrum of time series 'signal' as 1D array
    Frequencies in MHz (hence the 1.0e-6), assumed a 0.1 ns sample period
    """
    return (1.0e-6 * np.fft.rfftfreq(signal.shape[0], d=0.1e-9)) # MHz

def do_filter_signal_lowpass(signal, cutoff_freq):
    """
    For one signal time series, do lowpass filtering.

    Parameters
    ----------
    signal : time series, 1D array
    cutoff_freq : high frequency cutoff in MHz
    """
    freqs = get_freq_axis(signal)

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
    relative energy difference

    Parameters
    ----------
    test_signal : time series, 1D array
    orig_signal : idem
    upsampling_factor : upsampling factor for sub-sample timing accuracy, used to obtain optimal cross-correlation (optimized over arrival times). Default 10.
    """

    orig_signal_upsampled = scipy_signal.resample(orig_signal, upsampling_factor*len(orig_signal) )
    test_signal_upsampled = scipy_signal.resample(test_signal, upsampling_factor*len(test_signal) )

    #orig_upsampled = scipy.signal.resample(orig_signal, upsampling_factor*len(orig_signal) )
    #interp_upsampled = scipy.signal.resample(interpolated_signal, upsampling_factor*len(interpolated_signal) )

    crosscorr = scipy_signal.correlate(test_signal_upsampled, orig_signal_upsampled)
    #lags = signal.correlation_lags(orig_pulse.size, interpolated_pulse.size)
    #    lag = lags[np.argmax(crosscorr)]
    normalization = np.sqrt(np.sum(orig_signal_upsampled**2) * np.sum(test_signal_upsampled**2))
    crosscorr /= normalization

    autocorr = scipy_signal.correlate(orig_signal_upsampled, orig_signal_upsampled)
    max_autocorr = np.argmax(autocorr) # this is at "t=0"

    # Get Delta t, maximum at zero time shift, maximum overall
    CC_optimized_timeshift = np.max(crosscorr)
    CC_zeroshift = crosscorr[max_autocorr] # the CC value at fixed timing

    delta_t = 0.1 * (1.0 / upsampling_factor) * (np.argmax(crosscorr) - np.argmax(autocorr)) # 0.1 ns per sample in original signal

    # so delta_t is given in ns

    orig_energy = np.sum(orig_signal_upsampled**2)
    test_energy = np.sum(test_signal_upsampled**2)
    energy_rel_diff = (test_energy - orig_energy) / orig_energy

    return (CC_zeroshift, CC_optimized_timeshift, delta_t, energy_rel_diff)



def plot_pulse_and_spectrum(orig_time_axis, orig_pulse, interpolated_time_axis, interpolated_pulse, x, y, cutoff_freq, pol):
    """
    Plots an interpolated pulse together with a 'true' simulated pulse

    Parameters
    ----------
    orig_time_axis : np.ndarray
        Time axis for the original pulse
    orig_pulse : np.ndarray
        time trace, 1D array
    interpolated_pulse : np.ndarray
    interpolated_time_axis : np.ndarray
        Time axis for the interpolated pulse
    x : the x position (float), for annotation in the plot
    y : idem for y
    cutoff_freq : value of estimated cutoff frequency, for annotation only
    pol : polarization number
    """
    radius = np.sqrt(x**2 + y**2)
    freqs = get_freq_axis(orig_pulse)

    (CC_zeroshift, CC_optimized_timeshift, delta_t, energy_rel_diff) = get_crosscorrelation(orig_pulse, interpolated_pulse)

    fig, ax = plt.subplots(figsize=(10.67, 4), nrows=1, ncols=2)
    ax1, ax2 = ax[0], ax[1]

    #plt.figure()
    ax1.plot(orig_time_axis, 1.0e6 * orig_pulse, label='orig pulse', lw=2)
    ax1.plot(interpolated_time_axis, 1.0e6 * interpolated_pulse, label='interpolated pulse', lw=2)

    time_offset = int(orig_pulse.argmax() - interpolated_pulse.argmax())
    residual = 1.0e6 * (np.roll(interpolated_pulse, time_offset) - orig_pulse)
    ax1.plot(orig_time_axis, residual, label='difference', lw=2, c='g')

    interpolated_energy = np.sum(interpolated_pulse**2)
    orig_energy = np.sum(orig_pulse**2)
    #fixed_pulse = interpolated_pulse * np.sqrt(orig_energy / interpolated_energy)
    #plt.plot(time_axis, fixed_pulse, label='fixed pulse', c='r', lw=1)
    ax1.grid()
    ax1.set_xlabel('Time [ ns ]')
    ax1.set_ylabel(r'E-field [ $\mu$V/m ]')
    ax1.set_xlim(orig_time_axis[0], orig_time_axis[500])
    ax1.legend(loc='best')

    #plt.figure()
    orig_pulse_powerspec = np.abs(np.fft.rfft(orig_pulse))**2
    interp_pulse_powerspec = np.abs(np.fft.rfft(interpolated_pulse))**2
    ax2.plot(freqs, orig_pulse_powerspec, label='Orig pulse')
    ax2.plot(freqs, interp_pulse_powerspec, label='Interpolated pulse')
    ax2.text(0.98, 0.40, 'Position x = %3.1f, y = %3.1f, r = %3.2f m, pol = %d' % (x, y, radius, pol), transform=plt.gca().transAxes, ha='right') #, va='right')
    ax2.text(0.98, 0.30, 'CC = %1.5f, CC_max = %1.5f' % (CC_zeroshift, CC_optimized_timeshift), transform=plt.gca().transAxes, ha='right')
    ax2.text(0.98, 0.20, 'delta_t = %1.2f ns, cutoff freq = %3.1f MHz' % (delta_t, cutoff_freq), transform=plt.gca().transAxes, ha='right')

    #plt.yscale('log')
    ax2.grid()
    ax2.set_xlabel('Frequency [ MHz ]')
    ax2.set_ylabel('Power spectrum [ a.u. ]')
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0.0, 1.2*np.max(orig_pulse_powerspec))
    ax2.legend(loc='best')

    plt.show()


def read_data_hdf5(filename):
    """
    Reading in the demo data hdf5 file.
    The time traces inside are CoREAS E-fields, converted to 2 'on-sky' polarizations.
    """
    try:
        demo_file = h5py.File(filename, 'r')
    except:
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

    return (zenith, azimuth, xmax, footprint_pos_x, footprint_pos_y, test_pos_x, test_pos_y, footprint_antenna_data, test_antenna_data, footprint_time_axis, test_time_axis)
