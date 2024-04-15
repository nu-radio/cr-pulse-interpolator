# cr-pulse-interpolator
Full electric-field waveform interpolation for air shower simulations

For elaborate information on the methods and their performance, see  
Corstanje et al. (2023), JINST 18 P09005, arXiv **2306.13514**  
Please cite this when using code and/or method in your analysis, publication etc.

## Contents
The package contains two interpolation submodules:
- **interpolation_fourier**: interpolates a 2D scalar function f(x, y) defined on a radial (polar) grid.
  This includes for instance air shower radio energy fluence or amplitude along a radio footprint in the shower plane.
- **signal_interpolation_fourier**: interpolates full time traces (waveforms) of electric-field signals simulated by air shower simulation codes such as CoREAS, in antennas placed on a radial grid.

## Installation and Dependencies
This package can be installed with the following command:  
```sh
  pip install git+https://github.com/nu-radio/cr-pulse-interpolator
```
To automatically install the requirements for the demo scripts, one can instead use  
```sh
  pip install "cr_pulse_interpolator[demo] @ git+https://github.com/nu-radio/cr-pulse-interpolator"  
```

## Usage 
The interpolation modules have an interface very similar to e.g. Scipy's **interp1d** class.  
Runnable demo scripts are:
- **demo_interpolation_fourier.py**: a demo of the interpolation_fourier method, plotting a radio energy footprint heatmap.
- **demo_signal_interpolation.py**: demonstrates full signal interpolation, doing cross-correlations with true simulated signals on "random" test positions.
- **minimal_usage_demo.py**: a smaller demo of full signal interpolation, aimed at getting the user started quickly.

The full signal interpolation demos require example data files to be downloaded:  
run **download_demo_data.sh** before using the demo scripts.  
The h5py package is required to read the demo data.

The example data files contain E-field traces from a CoREAS shower, converted to two "on-sky" polarizations.
They can be given as input in any order when creating the interpolator object.
This is the recommended usage in the interpolator.  
NB. in the full pulse interpolator, the two polarizations should _not_ be aligned to vxB and vx(vxB), as the near-zero amplitudes along circles in the footprint lead to poor accuracy. In the interpolation for amplitude or fluence only, this is not an issue.
It is recommended to rotate the polarizations by 45 degrees when this happens (e.g. zenith showers, or close to north-south axis in general), and rotate back after interpolating.

Another recommendation is to test the accuracy of interpolation by simulating some 10 or 20 additional antennas placed at strategic positions 'in between' the radial-grid positions.
