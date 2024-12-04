# Minimal usage demo script of Fourier interpolation of pulse signals along simulated CR radio footprints
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2023

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cr_pulse_interpolator.signal_interpolation_fourier as sigF

import demo_helper

""" 
For using the interpolator, the x and y positions of the simulated antennas in the shower plane are needed,
along with the time traces for all these antennas and polarizations
The antennas need not be ordered by position

Shapes are 1D array for x, y (in meters)
antenna traces 'footprint_antenna_data' in shape (Nant, Nsamples, Npol), i.e., in this example (208, 4082, 2)
Additional antenna positions have been simulated ('test_antenna_data' and 'test_pos_x', 'test_pos_y') to test 
the interpolation accuracy.

When within < ~10 degrees from the North-South axis (including zenith), rotate the `on-sky` polarizations by 45 degrees
to avoid alignment with vx(vxB) and therefore having near-zero signals along circles in the footprint
"""
# Read in demo data
demo_filename = 'demo_shower.h5'
(zenith, azimuth, xmax, footprint_pos_x, footprint_pos_y, test_pos_x, test_pos_y, footprint_antenna_data, test_antenna_data, footprint_time_axis, test_time_axis) = demo_helper.read_data_hdf5(demo_filename)

"""
Initialize the interpolator object. It needs the antenna positions (x and y as 1D arrays) and their time traces as 
3D arrays (Nants, Nsamples, Npols).
By default the phase-interpolating method is "phasor" (see article)
"""
print('Initializing interpolator object...')
phase_method = "phasor" # other option is "timing" which takes longer to initialize (to check which performs better in use cases not yet tested)
signal_interpolator = sigF.interp2d_signal(footprint_pos_x, footprint_pos_y, footprint_antenna_data, verbose=False, phase_method=phase_method)

test_index = 124 # take one of the 250 test positions
pol = 0
this_x, this_y = test_pos_x[test_index], test_pos_y[test_index]
print('Interpolating pulse at position x = %3.2f, y = %3.2f m' % (this_x, this_y))

orig_pulse = test_antenna_data[test_index] # shape is (Nsamples, Npol)

"""
Call the interpolator object to obtain the interpolated pulse at the desired position
Return shape is (Nsamples, Npol)
Optionally, it can be low-pass filtered to an estimated reliable cutoff frequency
This is a reliable yet sometimes overly conservative estimate up to which frequency the interpolation is accurate
"""
interpolated_pulse, timings, _, _ = signal_interpolator(this_x, this_y, full_output=True)

"""
Because the trace start times were not provided during the interpolator initialisation, the returned timings
array only contains the timing offset induced by centering the pulse (which is the default behaviour). To 
compare to the original pulse, we need to roll the interpolated pulse back to have the same time axis.
"""
sample_offset = int(timings / signal_interpolator.sampling_period * -1)

# Make a plot of the trace and spectrum
demo_helper.plot_pulse_and_spectrum(orig_pulse[:, pol], np.roll(interpolated_pulse[:, pol], sample_offset),
                                    this_x, this_y, -1, pol)
