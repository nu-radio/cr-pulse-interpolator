# Module for Fourier interpolation of pulsed signals along simulated radio footprints of cosmic-ray air showers
# Author: A. Corstanje (a.corstanje@astro.ru.nl), 2023
#
# See article: arXiv:xxxx.xxxxx, doi xxx
# Please cite this when using code and/or methods in your analysis

import numpy as np
from scipy.signal import hilbert, resample
import interpolation_fourier as interpF

class interp2d_signal:

    def get_spectra(self, signals):
        """
        # Do FFT of 'signals', assumed shape (Nants, Nsamples, Npol) i.e. the time traces are along the second axis.
        # Produce absolute-amplitude spectrum and phase spectrum
        Parameters
        ----------
        signals : the input time traces
        """
        Nsamples = signals.shape[1]

        if self.verbose:
            print('Doing FFTs...', end=' ')
        all_antennas_spectrum = np.fft.rfft(signals, axis=1)
        abs_spectrum = np.abs(all_antennas_spectrum)
        phasespectrum = np.angle(all_antennas_spectrum)
        unwrapped_phases = np.unwrap(phasespectrum, axis=1, discont=0.7*np.pi)
        if self.verbose:
            print('done.')

        #antnr = 1555
        #print(pos_sim_UVW[antnr])
        freqs = np.fft.rfftfreq(Nsamples, d=self.sampling_period)
        freqs /= 1.0e6 # in MHz

        freqstep = freqs[1] - freqs[0]
        nof_freq_channels = len(np.where( (freqs < 500) )[0])
        if self.verbose:
            print('Frequency channel width %2.4f MHz, there are %d channels between 0 and 500 MHz' % (freqstep, nof_freq_channels))

        return (freqs, all_antennas_spectrum, abs_spectrum, phasespectrum, unwrapped_phases)

    def hilbert_envelope_timing(self, signals, lowfreq=30.0, highfreq=500.0, upsample_factor=10, sum_over_pol=True, do_hilbert_envelope=True):
        """
        Produce pulse arrival times from Hilbert envelope maxima (if do_hilbert_envelope) or from raw E-field maxima.
        Assumed shape for 'signals' is (Nant, Nsamples, Npols), i.e. time traces along second axis
        Filtering is done between lowfreq and highfreq in MHz, timing is done on filtered signals (after inverse-FFT)
        Option sum_over_pol is to sum the square of the Hilbert envelopes of each polarization, to get one arrival time over all polarizations (may help at low signal strength), default True.

        Parameters
        ----------
        signals : 3D array of shape (Nant, Nsamples, Npols)
        lowfreq : low-frequency cutoff, default 30.0 MHz
        highfreq : high-frequency cutoff, default 500.0 MHz
        upsample_factor : upsampling factor to use for sub-sample timing accuracy, default 10
        sum_over_pol : sum over polarizations if do_hilbert_envelope, default True
        do_hilbert_envelope : use Hilbert envelope for timing if True, else direct E-field maximum
        """
        (Nant, Nsamples, Npols) = signals.shape

        if self.verbose:
            print('Bandpass filtering %d to %d MHz' % (int(lowfreq), int(highfreq)))
        spectrum = np.fft.rfft(signals, axis=1)
        freqs = np.fft.rfftfreq(Nsamples, d=self.sampling_period)
        freqs /= 1.0e6 # in MHz
        filtering_out = np.where( (freqs < lowfreq) | (freqs > highfreq)) # or!

        spectrum[:, filtering_out, :] *= 0.0
        filtered_signals = np.fft.irfft(spectrum, axis=1)

        # Get strongest polarization
        power_per_pol = np.sum(np.sum(filtered_signals**2, axis=1), axis=0)
        strongest_pol = np.argmax(power_per_pol)
        if self.verbose:
            print('Strongest polarization is %d' % strongest_pol)

        timestep = self.sampling_period # in s
        if self.verbose:
            print('Upsampling by a factor %d' % upsample_factor)
        signals_upsampled = resample(filtered_signals, upsample_factor*Nsamples, axis=1)

        nof_samples = signals_upsampled.shape[1]
        signals_upsampled = np.roll(signals_upsampled, nof_samples//2, axis=1) # Put in the middle of the block, avoiding negative values in timing

        if do_hilbert_envelope:
            if self.verbose:
                print('Hilbert envelope')
            hilbert_envelope = np.abs(hilbert(signals_upsampled, axis=1))

            if sum_over_pol:
                hilbert_sum_over_pol = np.sqrt( np.sum(hilbert_envelope**2, axis=2) )

                pulse_timings = (np.argmax(hilbert_sum_over_pol, axis=1) - nof_samples//2) * (timestep/upsample_factor)
            else:
                pulse_timings_per_pol = (np.argmax(hilbert_envelope, axis=1) - nof_samples//2) * (timestep/upsample_factor)

        else:
            pulse_timings_per_pol = (np.argmax(signals_upsampled, axis=1) - nof_samples//2) * (timestep/upsample_factor)

        #pulse_timings = np.argmax(hilbert_envelope[:, :, 1], axis=1) * (timestep/upsample_factor)
        if self.verbose:
            print('Timings done')

        if do_hilbert_envelope and sum_over_pol:
            pulse_timings_per_pol = np.zeros( (Nant, Npols) )
            for pol in range(Npols):
                pulse_timings_per_pol[:, pol] += pulse_timings # want to do this without yucky for loop

        return pulse_timings_per_pol


    def phase_wrap(self, phases):
        """
        wrap 'phases' (float or any array shape) into interval (-pi, pi)
        """
        return ((phases + np.pi) % (2 * np.pi) - np.pi)

    def timing_corrected_phases(self, freqs, phase_spectrum, pulse_timings):
        """
        Take phase_spectrum as input
        Account for a linear function from the pulse_timings, i.e.
        according to delta_phi = 2 pi f delta_t
        Return as timing-corrected phase spectrum

        Parameters
        ----------
        freqs : frequency axis of FFTs
        phase_spectrum : 3D array of shape (Nants, Nphases, Npols) containing phase spectra
        pulse_timings : 2D array of shape (Nants, Npols) containing pulse timings
        """
        (Nants, Nphases, Npols) = phase_spectrum.shape

        freq_step = freqs[1]
        phase_spectrum_corrected = np.zeros( phase_spectrum.shape )

        for i, ant in enumerate(range(Nants)):
            for pol in range(Npols):

                this_phase_corrections = 2*np.pi * (freqs*1.0e6) * pulse_timings[i, pol] # plus or minus sign?
                phase_spectrum_corrected[ant, :, pol] = phase_spectrum[ant, :, pol] + this_phase_corrections # this can be done more efficiently...

        # WRAP phases into 0..2*pi
        phase_spectrum_corrected = self.phase_wrap(phase_spectrum_corrected)
        phase_spectrum_corrected = np.unwrap(phase_spectrum_corrected, axis=1, discont=0.7*np.pi)
        # needed?

        return phase_spectrum_corrected

    def phase_unwrap_2d(self, x, y, phases):
        """
        Basic method to unwrap phases in 2D
        The general problem of optimal 2D phase unwrapping is NP-complete (see ref in article), so this will only work for well-behaved, slowly varying phases
        First do 1D unwrap over radial directions
        Then 1D unwrap over angular directions (along circles in the radial grid)

        Parameters
        ----------
        x : 1D array of antenna position x (in any order)
        y : same for y (same order)
        phases : 1D array, i.e. one phase per antenna (same order of antennas as x, y)
        """
        # Put them into a 2D array with a radial and an angular axis
        indices = interpF.interp2d_fourier.get_ordering_indices(x, y)
        phases_ordered = phases[indices]

        # First along radial axis, then angular axis
        phases_unwrapped_2d = np.unwrap(phases_ordered, axis=0)
        phases_unwrapped_2d  = np.unwrap(phases_unwrapped_2d, axis=1)
        #phases_unwrapped_2d  = np.unwrap(phases_unwrapped_2d, axis=0)

        # Back into original 1D shape...
        phases_unwrapped = np.zeros( phases.shape )

        (Nradial, Nangular, Npol) = phases_unwrapped_2d.shape

        phases_unwrapped[indices.ravel()] = phases_unwrapped_2d.reshape( (Nradial*Nangular, Npol) )

        return phases_unwrapped

    def degree_of_coherency(self, low_freq=30.0, high_freq=500.0):
        """
        This implements Eq. (2.4) in the article, for given frequency band limits
        """
        complex_phases = np.exp(1.0j * self.phasespectrum_corrected)
        ampli = self.abs_spectrum

        spectrum_corrected = ampli * complex_phases

        freq_range = np.where( (self.freqs > low_freq) & (self.freqs < high_freq))[0]

        complex_sum = np.sum(spectrum_corrected[:, freq_range, :], axis=1)

        abs_sum = np.sum(ampli[:, freq_range, :], axis=1)

        coherency = np.abs(complex_sum) / abs_sum

        return coherency

    def get_constant_phases(self, low_freq=30.0, high_freq=500.0):
        """
        This implements Eq. (2.3) in the article, for given frequency band limits.
        Phases have been corrected to have maximum Hilbert envelope at "t"=0
        So, add up the complex phases, weighted by the amplitudes, to get the constant phase
        which determines if the pulse is cos-like or sin-like, or a value in between
        """
        complex_phases = np.exp(1.0j * self.phasespectrum_corrected)
        ampli = self.abs_spectrum

        spectrum_corrected = ampli * complex_phases

        freq_range = np.where( (self.freqs > low_freq) & (self.freqs < high_freq))[0]
        complex_sum = np.sum(spectrum_corrected[:, freq_range, :], axis=1)

        const_phases = np.angle(complex_sum)

        # Do unwrapping in 2D to avoid 2 pi periodicity mismatches (spurious jumps of 2 pi)
        const_phases_unwrapped = self.phase_unwrap_2d(self.pos_x, self.pos_y, const_phases)

        return const_phases_unwrapped

    def get_coherency_vs_frequency(self, bandwidth=50.0, low_freq=30.0, high_freq=500.0, coherency_cutoff=0.8):
        """
        Compute `degree of coherency` as from Eq. (2.4) in a sliding frequency window
        each with bandwidth 50 MHz (or optional value given)
        Then, see at which frequency this value first drops below 'coherency_cutoff'
        That frequency defines the 'cutoff frequency' (cutoff_freq) below which
        the signal is well approximated by a standard impulse, and thus reliably interpolated.

        Parameters
        ----------
        bandwidth : frequency bandwidth in MHz, default 50.0
        low_freq : low frequency cutoff in MHz, default 30.0
        high_freq : high frequency cutoff in MHz, default 500.0
        coherency_cutoff : threshold value for coherency level, default 0.8
        """
        freq_indices = np.where( (self.freqs > low_freq) & (self.freqs < (high_freq)))[0] # high_freq - bandwidth?

        (Nants, Nsamples, Npols) = self.phasespectrum_corrected.shape
        coherency_vs_freq = np.zeros( (Nants, len(freq_indices), Npols))

        cutoff_freq = np.zeros( (Nants, Npols) )

        for i, freq_index in enumerate(freq_indices):
            if self.verbose:
                print('%d / %d' % (i, len(freq_indices)))
            this_freq = self.freqs[freq_index]
            coherency = self.degree_of_coherency(low_freq=this_freq, high_freq=this_freq+bandwidth)
            coherency_vs_freq[:, i, :] = coherency

        # cutoff_values = np.where(coherency_vs_freq < 0.8)
        # TODO do this without / with fewer for loops
        # do explicitly in for loops for now...
        for ant in range(Nants):
            for pol in range(Npols):
                cutoff_index = np.where(coherency_vs_freq[ant, :, pol] < coherency_cutoff)[0] # first index where < 0.8
                if len(cutoff_index) > 0:
                    cutoff_index = cutoff_index[0]
                else:
                    cutoff_index = -1
                cutoff_freq[ant, pol] = min(self.freqs[freq_indices[cutoff_index]], high_freq) # max 500.0 cap, needed?

        return (coherency_vs_freq, cutoff_freq)

    def get_freq_dependent_timing_correction(self, lowfreq=30.0, highfreq=500.0, bandwidth=50.0, upsample_factor=10, ignore_cutoff_freq_in_timing=False):
        """
        Implements "method (2)" to account for phase spectra towards higher frequencies.
        Determines arrival time (E-field maximum) in a 50 MHz (or 'bandwidth') sliding frequency window, by filtering to each frequency window and inverse-FFT.
        Arrival time is mapped to phase in the center of each window.

        Parameters
        ----------
        low_freq : low frequency cutoff in MHz, default 30.0
        high_freq : high frequency cutoff in MHz, default 500.0
        bandwidth : frequency bandwidth in MHz, default 50.0
        upsample_factor : upsampling factor for timing accuracy, default 10
        ignore_cutoff_freq_in_timing: proceed with pulse timing beyond cutoff frequency, default False
        """
        start_freq = self.freqs[self.freqs>=lowfreq][0]
        end_freq = self.freqs[self.freqs<=(highfreq+bandwidth/2)][-1]

        freq_indices = np.where((self.freqs >= lowfreq) & (self.freqs < (highfreq+bandwidth/2)))[0]

        spectrum_after_correction_so_far = self.abs_spectrum * np.exp(1j * self.phasespectrum_corrected)
        signals = np.fft.irfft(spectrum_after_correction_so_far, axis=1)

        freq_dependent_timing = np.zeros( self.abs_spectrum.shape )

        for i, freq_index in enumerate(freq_indices):
            if i % 3 != 0:
                freq_dependent_timing[:, freq_index, :] += freq_dependent_timing[:, freq_index-1, :]
            else:
                this_freq = self.freqs[freq_index]

                band_low = max(lowfreq, this_freq - bandwidth/2)
                band_high = min(highfreq+bandwidth, this_freq + bandwidth/2)
                #band_low = np.max(lowfreq, this_freq - bandwidth/2)
                #band_high = min(highfreq+bandwidth/2, this_freq + bandwidth/2)
                #band_low = max(lowfreq, band_high - bandwidth)
                if ((band_high - band_low) < bandwidth) and (band_low + bandwidth < band_high):
                    band_high = band_low + bandwidth
                # check bandwidth
                if self.verbose:
                    print('Bandwidth %3.1f MHz, from %3.1f to %3.1f MHz' % (band_high-band_low, band_low, band_high))

                this_timing = self.hilbert_envelope_timing(signals, lowfreq=band_low, highfreq=band_high, upsample_factor=upsample_factor, do_hilbert_envelope=False)
                # returns timings for (Nant, Npol)
                freq_dependent_timing[:, freq_index, :] += this_timing # check index broadcasting

        # do cutoff fixing here
        # keep timing constant beyond frequency cutoff
        if not ignore_cutoff_freq_in_timing:
            (Nants, Nsamples, Npols) = freq_dependent_timing.shape
            for ant in range(Nants):
                for pol in range(Npols):

                    # Keep timing values constant for frequencies > local cutoff value
                    this_index = np.where(self.freqs >= self.cutoff_freq[ant, pol])[0][0]
                    freq_dependent_timing[ant, this_index:, pol] = freq_dependent_timing[ant, this_index-1, pol]

                    for i, index in enumerate(freq_indices):
                        if i == 0:
                            continue
                        # Just in case, a similar stopping criterion when successive timings get too `noisy`
                        if np.abs(freq_dependent_timing[ant, index, pol] - freq_dependent_timing[ant, index-1, pol]) > 0.5e-9:
                            freq_dependent_timing[ant, index:, pol] = freq_dependent_timing[ant, index-1, pol]
                            break

        return freq_dependent_timing

    def nearest_antenna_index(self, x, y, same_radius=False, tolerance=1.0):
        """
        Search for closest antenna at the same radius, within tolerance (m), when same_radius=True
        Otherwise find the nearest antenna to (x, y) and return its index
        The nearest antenna position is then referenced as self.pos_x[index], self.pos_y[index]

        Parameters
        ----------
        x : x position (float, single value)
        y : idem for y
        same_radius : consider only antennas at the same radius, +/- 'tolerance'
        tolerance : tolerance in m for same_radius search
        """
        if not same_radius:
            index = np.argmin( (self.pos_x - x)**2 + (self.pos_y - y)**2 )
        else: # a bit convoluted...
            radius_antennas = np.sqrt(self.pos_x**2 + self.pos_y**2)
            thisradius = np.sqrt(x**2 + y**2)

            indices_same_radius = np.where( (radius_antennas > (thisradius-tolerance)) & (radius_antennas < (thisradius+tolerance)) )[0]
            assert len(indices_same_radius) > 0
            ant_x = self.pos_x[indices_same_radius]
            ant_y = self.pos_y[indices_same_radius]

            index_min = np.argmin( (ant_x - x)**2 + (ant_y - y)**2 )

            index = indices_same_radius[index_min]

        return index

    def get_cutoff_freq(self, x, y, pol):
        """
        A getter for the cutoff frequency in polarization 'pol' interpolated to arbitrary position (x, y)

        Parameters
        ----------
        x : x position (float, single value)
        y : idem for y
        pol : polarization number (int, single value)
        """
        return self.interpolators_cutoff_freq[pol](x, y)


    def __init__(self, x, y, signals, lowfreq=30.0, highfreq=500.0, sampling_period=0.1e-9, phase_method="phasor", radial_method='cubic', upsample_factor=5, coherency_cutoff_threshold=0.9, ignore_cutoff_freq_in_timing=False, verbose=False):
        """
        Initialize a callable signal interpolator object

        Parameters
        ----------
        x : 1D array for the simulated antenna positions (x) in m
        y : idem for y
        signals : 3D array of shape (Nant, Nsamples, Npols) with the antennas indexed in the first axis, the time traces in the second axis, and the polarizations in the third.
        lowfreq: low-frequency limit, typically set to 30 MHz. If a higher low-frequency limit is desired, it may likely be better to keep it at 30 MHz here, and high-pass filter later.
        highfreq : high-frequency limit, default 500.0 MHz, adjustable to e.g. 80 MHz.
        sampling_period : the time between samples in the data, default 0.1e-9 seconds (0.1 ns)
        phase_method : the options for the phase interpolation are "phasor" and "timing", cf. pg 8 in the article ("Method (1)" vs "Method (2)"), default "phasor" (Method (1))
        radial_method : the interp1d method used for radially interpolating Fourier coefficients. Usually set to 'cubic' for cubic splines, in interpolation_fourier.
        upsample_factor : upsampling factor used for sub-sample timing accuracy. Default 5.
        coherency_cutoff_threshold : the value of Eq. (2.4) that defines the reliable high-frequency limit on each position. Used internally in the "timing" method, also available to the user via self.get_cutoff_freq(x, y, pol) above. Default 0.9.
        ignore_cutoff_freq_in_timing : can be set to True when experimenting with the "timing" method without its stopping criterion, default False.
        verbose : print info while initializing
        """
        self.nofcalls = 0
        self.verbose = verbose
        self.method = phase_method
        self.pos_x = x
        self.pos_y = y
        (Nants, Nsamples, Npols) = signals.shape # hard assumption, 3D...
        self.trace_length = Nsamples
        self.sampling_period = sampling_period
        if self.verbose: print('Setting sampling period to %1.1e seconds' % self.sampling_period)
        # Get the abs-amplitude and phase spectra from the time traces
        (self.freqs, all_antennas_spectrum, self.abs_spectrum, self.phasespectrum, self.unwrapped_phases) = self.get_spectra(signals)

        if self.verbose:
            print('Doing timings using hilbert envelope...')

        # Get pulse timings using Hilbert envelope
        self.pulse_timings = self.hilbert_envelope_timing(signals, lowfreq=30.0, highfreq=80.0, upsample_factor=upsample_factor)

        # Remove timings from phase spectra
        self.phasespectrum_corrected = self.timing_corrected_phases(self.freqs, self.phasespectrum, self.pulse_timings)

        # Get phase constant or `Hilbert phase`
        self.const_phases = self.get_constant_phases(high_freq=80.0)

        # Remove phase constant from phase spectra
        self.phasespectrum_corrected -= self.const_phases[:, np.newaxis, :] # from (Nant, Npol) to (Nant, Nsamples, Npol)
        self.phasespectrum_corrected_before_freq_dependent = np.copy(self.phasespectrum_corrected) # for testing / demo purposes

        # Get degree of coherency and reliable high-cutoff frequency
        if self.verbose:
            print('Getting coherency and freq cutoff')
        (self.coherency_vs_freq, self.cutoff_freq) = self.get_coherency_vs_frequency(coherency_cutoff=coherency_cutoff_threshold)
        if self.method == "timing":
            if verbose: print('Doing freq dependent timing...')
            # Get arrival times in a sliding frequency window
            self.freq_dependent_timing = self.get_freq_dependent_timing_correction(upsample_factor=upsample_factor, ignore_cutoff_freq_in_timing=ignore_cutoff_freq_in_timing)
            if verbose: print('Done freq dependent timing')
        else:
            # print('NOT doing freq dependent timing')
            self.freq_dependent_timing = np.zeros(self.phasespectrum_corrected.shape)

        # Remove sliding-window timings from phase spectra
        for i, ant in enumerate(range(Nants)):
            for pol in range(Npols):
                this_phase_corrections = 2*np.pi * (self.freqs*1.0e6) * self.freq_dependent_timing[ant, :, pol]
                self.phasespectrum_corrected[ant, :, pol] = self.phasespectrum_corrected[ant, :, pol] + this_phase_corrections # this can be done more efficiently...

        self.coherency = self.degree_of_coherency()

        """
        Produce interpolators for amplitude and phase spectrum
        Do the Fourier interpolation for each frequency bin < 500 MHz, and for each polarization, separately
        Note: an order of magnitude speed improvement should be obtainable by vectorizing the Fourier interpolator
        """
        nof_freq_channels = len(np.where( (self.freqs < 500) )[0]) # remove hardcoded number

        self.interpolators_abs_spectrum = np.empty( (Npols, nof_freq_channels), dtype=object)# [ [None]*nof_freq_channels ] * Npols

        """
        Create interpolators for the "phasors" for each frequency,
        i.e. exp(i phi(f)) = (cos(i phi), sin(i phi)) for each frequency
        """
        self.interpolators_cosphi = np.empty( (Npols, nof_freq_channels), dtype=object)
        self.interpolators_sinphi = np.empty( (Npols, nof_freq_channels), dtype=object)

        self.interpolators_freq_dependent_timing = np.empty( (Npols, nof_freq_channels), dtype=object)

        self.interpolators_timing = np.empty(Npols, dtype=object)

        self.interpolators_constphase = np.empty(Npols, dtype=object)

        self.interpolators_cutoff_freq = np.empty(Npols, dtype=object)

        if verbose: print('Creating %d interpolators total' % (3*Npols*nof_freq_channels + 3*Npols), end=' ')

        # Create and initialize the interpolators for all quantities
        for freq_channel in range(nof_freq_channels):
            for pol in range(Npols):
                self.interpolators_abs_spectrum[pol, freq_channel] = interpF.interp2d_fourier(x, y, self.abs_spectrum[:, freq_channel, pol])
                self.interpolators_freq_dependent_timing[pol, freq_channel] = interpF.interp2d_fourier(x, y, self.freq_dependent_timing[:, freq_channel, pol])

                self.interpolators_cosphi[pol, freq_channel] = interpF.interp2d_fourier(x, y, np.cos(self.phasespectrum_corrected[:, freq_channel, pol]) )
                self.interpolators_sinphi[pol, freq_channel] = interpF.interp2d_fourier(x, y, np.sin(self.phasespectrum_corrected[:, freq_channel, pol]) )


        for pol in range(Npols):
            self.interpolators_timing[pol] = interpF.interp2d_fourier(x, y, self.pulse_timings[:, pol])
            self.interpolators_constphase[pol] = interpF.interp2d_fourier(x, y, self.const_phases[:, pol])
            self.interpolators_cutoff_freq[pol] = interpF.interp2d_fourier(x, y, self.cutoff_freq[:, pol])

        if self.verbose:
            print('Done.')
    # end __init__

    def __call__(self, x, y, lowfreq=30.0, highfreq=500.0, filter_up_to_cutoff=False, account_for_timing=True, const_time_offset=20.0e-9, full_output=False):
        """
        Call the object, which computes the interpolation at arbitrary position (x, y)

        Parameters
        ----------
        x : the x position in m (float, single value)
        y : idem for y
        lowfreq : low-frequency limit for bandpass filtering of interpolated pulse, default 30.0 MHz
        highfreq : high-frequency limit, idem, default 500.0 MHz
        filter_up_to_cutoff : set to True for low-pass filtering up to local estimated cutoff frequency, default False
        account_for_timing : set to False to have each pulse at a fixed time given by 'const_time_offset' instead of its natural arrival time. Default True
        const_time_offset : constant time offset if not using interpolated arrival times. Default 20e-9 (seconds).
        full_output : set to True to output both time series and spectra. Default False, returns only time series.
        """
        if (self.nofcalls == 0) and self.verbose:
            print('Method: %s' % self.method)
        self.nofcalls += 1

        Nfreqs = len(self.interpolators_abs_spectrum[0])
        Npols = len(self.interpolators_abs_spectrum)

        freqs = np.fft.rfftfreq(self.trace_length, d=self.sampling_period)
        freqs /= 1.0e6 # in MHz
        ## Make self.freqs (todo)

        # Set up reconstructed spectra, timings, phases at freq=0
        abs_spectrum = np.zeros( (self.trace_length//2+1, Npols) )
        phasespectrum = np.zeros( (self.trace_length//2+1, Npols) )
        timings = np.zeros(Npols)
        const_phases = np.zeros(Npols)

        if self.method == 'timing':
            index_nearest = self.nearest_antenna_index(x, y)
            print('pos x = %3.2f, y = %3.2f: nearest antenna at x = %3.2f, y = %3.2f m' % (x, y, self.pos_x[index_nearest], self.pos_y[index_nearest]))

            phasespectrum = np.copy(self.phasespectrum_corrected[index_nearest]) # COPY !!!
            """
            Do nearest-neighbor interpolation on remaining ('corrected') phases, which should be near zero, but do full interpolation on amplitude spectrum
            """
            for freq_channel in range(Nfreqs):
                for pol in range(Npols): # todo: reduce
                    thisPower = self.interpolators_abs_spectrum[pol, freq_channel](x, y)
                    #thisPhase = self.interpolators_phase[pol, freq_channel](x, y)
                    abs_spectrum[freq_channel, pol] = thisPower

            # Account for freq dependent timings, which are interpolated first to (x, y)
            for freq_channel in range(Nfreqs):
                for pol in range(Npols):

                    this_timing = self.interpolators_freq_dependent_timing[pol, freq_channel](x, y)
                    this_phaseshift = -1.0e6*self.freqs[freq_channel] * 2*np.pi * this_timing
                    phasespectrum[freq_channel, pol] += this_phaseshift


        elif self.method == 'phasor':
            for freq_channel in range(Nfreqs):
                for pol in range(Npols):
                    # Interpolate abs-amplitude spectrum and phasors
                    thisPower = self.interpolators_abs_spectrum[pol, freq_channel](x, y)
                    this_realpart = self.interpolators_cosphi[pol, freq_channel](x, y)
                    this_imagpart = self.interpolators_sinphi[pol, freq_channel](x, y)

                    thisPhase = np.angle(this_realpart + 1.0j*this_imagpart)
                    # making unit vector by dividing by abs(re**2 + im**2) and multiplying that may be significantly faster

                    abs_spectrum[freq_channel, pol] = thisPower
                    phasespectrum[freq_channel, pol] = thisPhase

        else:
            raise ValueError('Unknown reconstruction method: %s' % self.method)

        # Apply the 30-80 MHz arrival times and phase constants, each interpolated to (x, y) first
        for pol in range(Npols):
            timings[pol] = self.interpolators_timing[pol](x, y)
            const_phases[pol] = self.interpolators_constphase[pol](x, y)

            # Account for timing
            if account_for_timing:
                phase_shifts = -1.0e6*freqs * 2*np.pi * timings[pol]
                phasespectrum[:, pol] += phase_shifts
            else:
                phase_shifts = -1.0e6*freqs * 2*np.pi * const_time_offset
                phasespectrum[:, pol] += phase_shifts

            # Account for constant phase
            phasespectrum[:, pol] += const_phases[pol]

        # Wrap into (-pi, pi) where needed, to tidy up
        phasespectrum = self.phase_wrap(phasespectrum)
        """
        Set frequency channels with negative abs-amplitude to 0. This can arise sometimes when Fourier-interpolating abs-amplitudes around a circle. Throw warning when this is needed.
        """
        indices_negative = np.where(abs_spectrum < 0)
        nof_negative = len(indices_negative[0])
        nof_negative_pol0 = len(np.where(indices_negative[1]==0)[0])

        if (nof_negative > 0):
            print('warning: negative values in abs_spectrum found: %d times. Setting to zero.' % nof_negative)
            abs_spectrum[indices_negative] = 0.0
        """
        Filter to bandwidth up to local cutoff frequency if desired, otherwise up to high frequency limit
        """
        for pol in range(Npols):
            high_cutoff = self.interpolators_cutoff_freq[pol](x, y) if filter_up_to_cutoff else highfreq

            filter_indices = np.where( (freqs<lowfreq) | (freqs>high_cutoff))
            abs_spectrum[filter_indices, pol] *= 0.0

        # Produce reconstructed spectrum and time series
        spectrum = abs_spectrum * np.exp(1.0j * phasespectrum)

        timeseries = np.fft.irfft(spectrum, axis=0)

        if full_output:
            return (timeseries, abs_spectrum, phasespectrum)
        else:
            return timeseries
