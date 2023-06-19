# cr-pulse-interpolator
Full waveform interpolation for air shower simulations

## Important info
For elaborate information on the methods and their performance, see 
Corstanje et al. (2023), arXiv xxxx.xxxxx, doi xx.xxxx,
Please cite this when using code and/or method in your analysis, publication etc.

## Contents
The package contains two interpolation modules:
- interpolation_Fourier.py: interpolates a 2D scalar function f(x, y) defined on a radial (polar) grid.
  This includes for instance air shower radio energy fluence or amplitude along a radio footprint in the shower plane.
- signal_interpolation_Fourier.py: interpolates full time traces (waveforms) of electric-field signals simulated by air shower simulation codes such as CoREAS, in antennas placed on a radial grid.

## Usage 
The interpolation modules have an interface very similar to e.g. Scipy's **interp1d** class.
Runnable demo scripts are:
- demo_interpolation_fourier.py: a demo of the interpolation_fourier method, plotting a radio energy footprint heatmap.
- demo_signal_interpolation.py: demonstrates full signal interpolation, doing cross-correlations with true simulated signals on "random" test positions.
- minimal_usage_demo.py: a smaller demo of full signal interpolation, aimed at getting the user started quickly.

The full signal interpolation demos require example data files to be downloaded:
run **download_demo_data.sh** before using the demo scripts.

The example data files contain E-field traces from a CoREAS shower, converted to two "on-sky" polarizations.
This is the recommended usage in the interpolator.
NB. the two polarizations should _not_ be aligned to vxB and vx(vxB), as the near-zero amplitudes along circles in the footprint lead to poor accuracy.
It is recommended to rotate the polarizations by 45 degrees when this happens (e.g. zenith showers), and rotate back after interpolating.

Another recommendation is to test the accuracy of interpolation by simulating some 10 or 20 additional antennas placed at strategic positions 'in between' the radial-grid positions.
