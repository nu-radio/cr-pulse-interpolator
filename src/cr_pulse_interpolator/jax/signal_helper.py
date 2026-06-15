import jax
import jax.numpy as jnp

def hilbert(x : jax.typing.ArrayLike, N = None, axis : int=0) -> jax.Array:
    """
    The Hilbert envelope of the signal. 

    This is a simple re-implementation of `scipy.signal.hibert` which currently
    does not exist in the JAX API. 

    Parameter:
    ---------
    x : jax.typing.ArrayLike
        The signal to take the Hilbert envelope on
    axis : int, default=0
        The axis in which the signal is defined.

    Return:
    -------
    x_hilbert : jax.Array
        The hilbert envelope of the signal
    """
    x = jnp.asarray(x)
    if jnp.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = jnp.fft.fft(x, N, axis=axis)
    h = jnp.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h = h.at[0].set(1)
        h = h.at[N // 2].set(1)
        h = h.at[1:N // 2].set(2)
    else:
        h = h.at[0].set(1)
        h = h.at[1:(N+1) // 2].set(2)

    if x.ndim > 1:
        ind = [None] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = jnp.fft.ifft(Xf * h, axis=axis)
    return x

def resample(
    x: jax.typing.ArrayLike,
    n_resamples : int,
    axis: int = 0,
) -> jax.Array:
    """
    Resample the trace by the number of desired samples.

    This is jaxified from scipy.signal.resample.
    We also remove the functionality for complex signals since only
    real signal are applicable in our scenario.

    Parameter:
    ----------
    x : jax.typing.ArrayLike
        the trace to resample
    n_resample : int
        the number of samples to re-sample to
    axis : int, default=0
        the axis where the trace is to downsample

    Return:
    ------
    y : jax.Array
        the downsampled signal
    """
    x = jnp.asarray(x)
    n_samples = x.shape[axis]
    trace_fft = jnp.fft.rfft(x, axis=axis)

    # print(f"Corresponding to reduction of {n_samples:d} to {n_resamples:d} number of samples.")

    # Placeholder array for output spectrum
    newshape = list(trace_fft.shape)
    newshape[axis] = n_resamples // 2 + 1
    Y = jnp.zeros(newshape, trace_fft.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(n_resamples, n_samples)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y = Y.at[tuple(sl)].set(trace_fft[tuple(sl)])


    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if n_resamples < n_samples:  # downsampling
            sl[axis] = slice(N//2, N//2 + 1)
            Y = Y.at[tuple(sl)].set(Y[tuple(sl)] * 2.)
        elif n_samples < n_resamples:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y = Y.at[tuple(sl)].set(Y[tuple(sl)] * 0.5)

    # Inverse transform
    y = jnp.fft.irfft(Y, n_resamples, axis=axis)
    y *= (float(n_resamples) / float(n_samples))

    return y