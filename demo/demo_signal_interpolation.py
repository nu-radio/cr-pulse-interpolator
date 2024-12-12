# Demo script of Fourier interpolation of pulse signals along simulated CR radio footprints
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2023

import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

import cr_pulse_interpolator.signal_interpolation_fourier as sigF 

import demo_helper

"""
For using the interpolator, the x and y positions of the simulated antennas in the shower plane are needed,
along with the time traces for all these antennas and polarizations
Shapes are 1D array for x, y (in meters)
antenna traces in shape (Nant, Nsamples, Npol), i.e., in this example (208+250, 4082, 2)
"""

demo_filename = 'demo_shower.h5'
(zenith, azimuth, xmax, footprint_pos_x, footprint_pos_y, test_pos_x, test_pos_y, footprint_antenna_data, test_antenna_data, footprint_time_axis, test_time_axis) = demo_helper.read_data_hdf5(demo_filename)

plt.figure()
plt.scatter(footprint_pos_x, footprint_pos_y, c='b', marker='s')
plt.scatter(test_pos_x, test_pos_y, c='r', marker='x')
plt.gca().set_aspect('equal')
plt.xlabel('Meters vxB')
plt.ylabel('Meters vx(vxB)')

nof_test_positions = test_pos_x.shape[0] # the number of test antennas, here 250
azimuth_deg = (azimuth % (2*np.pi)) * 180.0/np.pi
azimuth_deg_clockwise_from_north = 90.0 - azimuth_deg
zenith_deg = zenith * 180.0/np.pi

print('Shower data from a 10^17 proton, azimuth = %3.1f, zenith = %3.1f deg, Xmax = %4.2f g/cm2' % (azimuth_deg_clockwise_from_north, zenith_deg, xmax))
print('(azimuth as clockwise from North)')

test_radius = np.sqrt(test_pos_x**2 + test_pos_y**2) # the core distance of each of the test antennas

phase_method = "phasor" # other option is "timing"
if phase_method == "timing":
    print('Initializing interpolator, this may take 1 to 3 minutes...')
else:
    print('Initializing interpolator...')

signals_start_times = footprint_time_axis[:, 0]
print(signals_start_times.shape)

signal_interpolator = sigF.interp2d_signal(footprint_pos_x, footprint_pos_y, footprint_antenna_data, verbose=True, phase_method=phase_method, signals_start_times=signals_start_times)
print('Done.')

test_indices = (23, 124, 20, 34) # Evaluate interpolation at these test positions
pol = 0 # polarization '0' for demo plots
for index in test_indices:
    this_x, this_y = test_pos_x[index], test_pos_y[index]
    print('Interpolating pulse at position x = %3.2f, y = %3.2f m' % (this_x, this_y))

    orig_pulse = test_antenna_data[index]

    interpolated_pulse, timings, _, _ = signal_interpolator(this_x, this_y, full_output=True, pulse_centered=False)
    #sample_offset = int(timings / signal_interpolator.sampling_period * -1)

    orig_pulse = orig_pulse[:, pol]
    #interpolated_pulse = np.roll(interpolated_pulse[:, pol], sample_offset)  # do only strongest polarization
    interpolated_pulse = interpolated_pulse[:, pol] # instead of the roll

    this_cutoff_freq = signal_interpolator.get_cutoff_freq(this_x, this_y, pol)

    #filtered_orig = demo_helper.do_filter_signal_lowpass(orig_pulse, this_cutoff_freq)

    (CC_zeroshift, CC_optimized_timeshift, delta_t, energy_rel_diff) = demo_helper.get_crosscorrelation(orig_pulse, interpolated_pulse)
    print('Normalized cross correlation (CC) = %1.4f, time mismatch = %1.3f ns' % (CC_zeroshift, delta_t))

    interpolated_time_axis = np.arange(len(interpolated_pulse)) * signal_interpolator.sampling_period + timings
    demo_helper.plot_pulse_and_spectrum(
        test_time_axis[index], orig_pulse,
        interpolated_time_axis, interpolated_pulse,
        this_x, this_y, this_cutoff_freq, pol
    )

"""
Evaluate accuracy of arrival (start) time per antenna
for all 250 test positions
"""

core_distances = np.zeros(nof_test_positions)
time_mismatches = np.zeros(nof_test_positions)

for index in range(nof_test_positions):
    this_x, this_y = test_pos_x[index], test_pos_y[index]
    core_distance = np.sqrt(this_x**2 + this_y**2)
    print('Interpolating pulse at position x = %3.2f, y = %3.2f m' % (this_x, this_y))

    real_start_time = test_time_axis[index][0]

    interpolated_pulse, interpolated_start_time, _, _ = signal_interpolator(this_x, this_y, full_output=True, pulse_centered=False)
    
    timing_mismatch = interpolated_start_time - real_start_time
    timing_mismatch *= 1.0e9 # ns 

    time_mismatches[index] = timing_mismatch
    core_distances[index] = core_distance
    print(f'Core distance = {core_distance:3.2f} m: Time mismatch = {timing_mismatch:3.3f} ns')

    #sample_offset = int(timings / signal_interpolator.sampling_period * -1)

plt.figure()
plt.scatter(core_distances, time_mismatches)
plt.xlabel('Core distance [ m ]')
plt.ylabel('Start time mismatch [ ns ]')


"""
Evaluate cross-correlation between true and interpolated pulses
for all 250 test positions, and for each of the 2 `on-sky' polarizations
"""
CC_values = np.zeros( (nof_test_positions, 2) )
distances = np.zeros(nof_test_positions)
for index in range(nof_test_positions):
    this_x, this_y = test_pos_x[index], test_pos_y[index]
    core_distance = np.sqrt(this_x**2 + this_y**2)
    print('Interpolating pulse at position x = %3.2f, y = %3.2f m' % (this_x, this_y))

    orig_pulse = test_antenna_data[index]

    interpolated_pulse, timings, _, _ = signal_interpolator(this_x, this_y, full_output=True, pulse_centered=False)
    sample_offset = 0 # int(timings / signal_interpolator.sampling_period * -1)

    for pol in (0, 1):
        this_cutoff_freq = signal_interpolator.get_cutoff_freq(this_x, this_y, pol)

        #filtered_orig = demo_helper.do_filter_signal_lowpass(orig_pulse, this_cutoff_freq)

        (CC_zeroshift, CC_optimized_timeshift, delta_t, energy_rel_diff) = demo_helper.get_crosscorrelation(
            orig_pulse[:, pol], np.roll(interpolated_pulse[:, pol], sample_offset)
        )
        print('Normalized cross correlation (CC) = %1.4f, time mismatch = %1.3f ns' % (CC_zeroshift, delta_t))

        CC_values[index, pol] = CC_zeroshift
        distances[index] = core_distance

print('\n\n')
print(f'Start time mismatches stddev (i.e. timing error) = {np.std(time_mismatches):3.4f} ns')

plt.figure()
plt.scatter(distances, CC_values[:, 0], label='pol 0')
plt.scatter(distances, CC_values[:, 1], label='pol 1')
plt.xlabel('Core distance [ m ]')
plt.ylabel('Normalized CC')
plt.grid()
plt.legend(loc='best')
plt.show()
