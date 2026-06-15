# Demo script of Fourier interpolation of pulse signals along simulated CR radio footprints
# JAX version — adapted from demo/demo_signal_interpolation.py
#
# Key differences from the NumPy version:
#   - ordered_indices must be precomputed and passed to the constructor
#   - initialisation is split into __init__ (shape / parameters) and initialize (computation)
#   - all test positions are evaluated in a single batched call
#   - output is a dict: result['traces'], result['start_times'], result['abs_spectrum'], ...
#   - traces are always pulse-centred; the original CoREAS traces are rolled to match

import os
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import cr_pulse_interpolator.jax.signal_interpolation_fourier as jax_sigF
import demo_helper

"""
Same data as the NumPy demo:
  footprint antennas  — used to initialise the interpolator
  test antennas       — used to evaluate accuracy
Shapes: (Nant, Nsamples, Npol) for traces, (Nant, Nsamples) for time axes
"""

demo_filename = os.path.join(os.path.dirname(__file__), '..', 'demo_shower.h5')
fig_path = os.path.join(os.path.dirname(__file__), 'demo_images_signal')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
(zenith, azimuth, xmax,
 footprint_pos_x, footprint_pos_y,
 test_pos_x, test_pos_y,
 footprint_antenna_data, test_antenna_data,
 footprint_time_axis, test_time_axis) = demo_helper.read_data_hdf5(demo_filename)

plt.figure()
plt.scatter(footprint_pos_x, footprint_pos_y, c='b', marker='s', label='footprint')
plt.scatter(test_pos_x, test_pos_y, c='r', marker='x', label='test')
plt.gca().set_aspect('equal')
plt.xlabel('Meters vxB')
plt.ylabel('Meters vx(vxB)')
plt.legend(loc='best')
plt.savefig(os.path.join(fig_path, 'antenna_positions.png'), dpi=300, bbox_inches='tight')

nof_test_positions = test_pos_x.shape[0]
azimuth_deg = (azimuth % (2 * np.pi)) * 180.0 / np.pi
azimuth_deg_clockwise_from_north = 90.0 - azimuth_deg
zenith_deg = zenith * 180.0 / np.pi

print('Shower data from a 10^17 proton, azimuth = %3.1f, zenith = %3.1f deg, Xmax = %4.2f g/cm2' % (
    azimuth_deg_clockwise_from_north, zenith_deg, xmax))
print('(azimuth as clockwise from North)')

"""
Initialise the JAX signal interpolator.

Step 1 — __init__: store shape, ordered_indices, and frequency parameters.
Step 2 — initialize: compute all FFT spectra, pulse timings, and phase corrections.
"""
print('Initializing JAX interpolator...')

ordered_indices = demo_helper.get_ordered_indices(footprint_pos_x, footprint_pos_y)

signal_interpolator = jax_sigF.interp2d_signal(
    signal_shape=footprint_antenna_data.shape,
    ordered_indices=ordered_indices,
)
signal_interpolator.initialize(
    x=footprint_pos_x,
    y=footprint_pos_y,
    signals=footprint_antenna_data,
    signal_start_times=footprint_time_axis[:, 0],
)
print('Done.')

"""
Evaluate all test positions in a single batched call.
The JAX interpolator is JIT-compiled, so the first call triggers tracing/compilation;
subsequent calls are fast.
"""
print('Evaluating all %d test positions...' % nof_test_positions)
result = signal_interpolator(test_pos_x, test_pos_y)
# result['traces']:       (Ntest, Nsamples, Npol)  pulse-centred interpolated traces
# result['start_times']:  (Ntest,)                 start time of each centred trace
# result['abs_spectrum']: (Ntest, Nfreq, Npol)     interpolated amplitude spectrum
# result['phase_spectrum']:(Ntest, Nfreq, Npol)    interpolated phase spectrum
print('Done.')

sampling_period = signal_interpolator.sampling_period
Nsamples = result['traces'].shape[1]

"""
Plot a few individual interpolated pulses.

JAX always centres the pulse in the trace; the original CoREAS traces are not centred.
We roll the originals by Nsamples//2 so both traces align on the same time axis.
"""
test_indices = (23, 124, 20, 34)
pol = 0

for index in test_indices:
    this_x, this_y = test_pos_x[index], test_pos_y[index]
    print('Interpolating pulse at position x = %3.2f, y = %3.2f m' % (this_x, this_y))

    interp_trace = np.asarray(result['traces'][index, :, pol])

    # Absolute time axis for the centred interpolated trace
    start_time = float(result['start_times'][index])
    time_axis = start_time + np.arange(Nsamples) * sampling_period

    # Roll original trace to centre it, matching the JAX output convention
    orig_trace_centered = np.roll(test_antenna_data[index, :, pol], Nsamples // 2)

    (_, CC_optimized, delta_t, _) = demo_helper.get_crosscorrelation(interp_trace, orig_trace_centered)
    print('Normalised CC (optimised over dt) = %1.4f, time mismatch = %1.3f ns' % (CC_optimized, delta_t))

    demo_helper.plot_pulse_and_spectrum(
        time_axis, orig_trace_centered,
        time_axis, interp_trace,
        this_x, this_y, pol, fig_path=fig_path
    )

"""
Evaluate accuracy of interpolated arrival (start) times for all 250 test positions.

JAX centres the trace, so result['start_times'] is shifted back by Nsamples//2 * sampling_period
relative to the CoREAS start time. We undo this offset before comparing.
"""
core_distances = np.zeros(nof_test_positions)
time_mismatches = np.zeros(nof_test_positions)

for index in range(nof_test_positions):
    this_x, this_y = test_pos_x[index], test_pos_y[index]
    real_start_time = test_time_axis[index][0]

    # Recover the uncentred start time from the JAX result
    jax_start_time = float(result['start_times'][index]) + Nsamples // 2 * sampling_period

    timing_mismatch = (jax_start_time - real_start_time) * 1.0e9  # ns
    time_mismatches[index] = timing_mismatch
    core_distances[index] = np.sqrt(this_x**2 + this_y**2)
    print(f'Core distance = {core_distances[index]:3.2f} m: Time mismatch = {timing_mismatch:3.3f} ns')

print('\nStart time mismatches stddev (timing error) = %3.4f ns' % np.std(time_mismatches))

plt.figure()
plt.scatter(core_distances, time_mismatches)
plt.xlabel('Core distance [ m ]')
plt.ylabel('Start time mismatch [ ns ]')
plt.savefig(os.path.join(fig_path, 'timing_mismatch_vs_distance.png'), dpi=300, bbox_inches='tight')

"""
Evaluate cross-correlation between true and interpolated pulses
for all 250 test positions and both polarizations.

The original traces are rolled by Nsamples//2 to match the centred JAX output.
CC_optimized_timeshift is used so any residual timing offset does not penalise the CC.
"""
CC_values = np.zeros((nof_test_positions, 2))
distances = np.zeros(nof_test_positions)

for index in range(nof_test_positions):
    this_x, this_y = test_pos_x[index], test_pos_y[index]
    distances[index] = np.sqrt(this_x**2 + this_y**2)

    for pol in (0, 1):
        orig_trace_centered = np.roll(test_antenna_data[index, :, pol], Nsamples // 2)
        interp_trace = np.asarray(result['traces'][index, :, pol])
        (_, CC_optimized, _, _) = demo_helper.get_crosscorrelation(interp_trace, orig_trace_centered)
        CC_values[index, pol] = CC_optimized

print('\n')
print('Mean CC pol 0 = %1.4f, pol 1 = %1.4f' % (np.mean(CC_values[:, 0]), np.mean(CC_values[:, 1])))

plt.figure()
plt.scatter(distances, CC_values[:, 0], label='pol 0')
plt.scatter(distances, CC_values[:, 1], label='pol 1')
plt.xlabel('Core distance [ m ]')
plt.ylabel('Normalized CC')
plt.grid()
plt.legend(loc='best')
# plt.show()
plt.savefig(os.path.join(fig_path, 'CC_vs_distance.png'), dpi=300, bbox_inches='tight')
