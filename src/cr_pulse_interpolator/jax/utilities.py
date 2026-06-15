import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Any, Type, Union
import interpax
from functools import partial

def linear_interp(x, xp, fp):   
    return interpax.interp1d(xq=x, f=fp, x=xp, method='linear', extrap=True)

def cubic_interp(x, xp, fp):
    return interpax.interp1d(xq=x, f=fp, x=xp, method='cubic', extrap=True)

def cubic_interp_amplitude(x, xp, fp):
    return interpax.interp1d(xq=x, f=fp, x=xp, method='cubic', extrap=(fp[0], 0.0))
    

def batched_fourier_interp_1d(rad, rad_grid, fft_grid):
    """
    Batched linear interpolation with boundary handling, specifically for Fourier components.

    For left and right values, the values at the boundaries are used.

    Parameters
    ----------
    rad : jnp.ndarray
        Input points to interpolate. Shape: (Nrad).
    xp : jnp.ndarray
        Points to interpolate at. Shape: (Nrad_grid).
    fp : jnp.ndarray
        Function values at xp. Shape: (Nrad_grid, Nphi).

    Returns
    -------
    jnp.ndarray
        Interpolated values. Shape: (Nrad, Nphi).
    """
    _, Nphi = fft_grid.shape

    # Function to interpolate over angular arms
    def interp_fn(phi_idx):
        values = fft_grid[:,phi_idx]       # shape: (Ngrid,)
        return jax.vmap(cubic_interp, in_axes=(0, None, None))(rad, rad_grid, values)  # → (Nant,)

    # Vectorize over angular arms
    interp_vals = jax.vmap(lambda p: interp_fn(p))(jnp.arange(Nphi))  # shape: (Nrad, Nphi)

    # Reorder to (Nrad, Nphi
    return interp_vals.T

def batched_fourier_signal_interp(rad, rad_grid, fft_grid):
    """
    Batched linear interpolation with boundary handling, specifically for Fourier components.

    For left and right values, the values at the boundaries are used.

    This is different from the 1D case, where only fluence is considered. Here, the batched interpolator is 
    designed to handle signals, involving different antenna, frequency, and polarizations.

    Parameters
    ----------
    rad : jnp.ndarray
        Input points to interpolate. Shape: (Nrad).
    xp : jnp.ndarray
        Points to interpolate at. Shape: (Nrad_grid).
    fp : jnp.ndarray
        Function values at xp. Shape: (Nrad_grid, Nphi, Nfreq, Npol).

    Returns
    -------
    jnp.ndarray
        Interpolated values. Shape: (Nrad, Nphi, Nfreq, Npol).
    """
    Nrad_grid, Nphi, Nfreq, Npol = fft_grid.shape

    # Function to interpolate over 1 slice and 1 freq
    def interp_fn(freq_idx, pol_idx, phi_idx):
        values = fft_grid[:,phi_idx, freq_idx, pol_idx]       # shape: (Ngrid,)
        return jax.vmap(cubic_interp, in_axes=(0, None, None))(rad, rad_grid, values)  # → (Nant,)

    # Generate all index combinations
    phi_idces = jnp.arange(Nphi)
    freq_idces = jnp.arange(Nfreq)
    pol_idces = jnp.arange(Npol)

    # Vectorize over frequency and slice
    interp_vals = jax.vmap(
        lambda f : jax.vmap(
            lambda pol : jax.vmap(
                lambda phi: interp_fn(f,pol,phi)
            )(phi_idces)
          )(pol_idces)
      )(freq_idces)  # shape: (Nrad, Nphi, Nfreq, Npol)

    # Reorder to (Nrad, Nphi, Nfreq, Npol)
    return jnp.transpose(interp_vals, (3,2,0,1))

def batched_fourier_amplitude_interp(rad, rad_grid, fft_grid):
    """
    Batched linear interpolation with boundary handling, specifically for Fourier components.

    For left and right values, the values at the boundaries are used.

    This is different from the 1D case, where only fluence is considered. Here, the batched interpolator is 
    designed to handle signals, involving different antenna, frequency, and polarizations.

    Parameters
    ----------
    rad : jnp.ndarray
        Input points to interpolate. Shape: (Nrad).
    xp : jnp.ndarray
        Points to interpolate at. Shape: (Nrad_grid).
    fp : jnp.ndarray
        Function values at xp. Shape: (Nrad_grid, Nphi, Nfreq, Npol).

    Returns
    -------
    jnp.ndarray
        Interpolated values. Shape: (Nrad, Nphi, Nfreq, Npol).
    """
    Nrad_grid, Nphi, Nfreq, Npol = fft_grid.shape

    # Function to interpolate over 1 slice and 1 freq
    def interp_fn(freq_idx, pol_idx, phi_idx):
        values = fft_grid[:,phi_idx, freq_idx, pol_idx]       # shape: (Ngrid,)
        return jax.vmap(cubic_interp, in_axes=(0, None, None))(rad, rad_grid, values)  # → (Nant,)

    # Generate all index combinations
    phi_idces = jnp.arange(Nphi)
    freq_idces = jnp.arange(Nfreq)
    pol_idces = jnp.arange(Npol)

    # Vectorize over frequency and slice
    interp_vals = jax.vmap(
        lambda f : jax.vmap(
            lambda pol : jax.vmap(
                lambda phi: interp_fn(f,pol,phi)
            )(phi_idces)
          )(pol_idces)
      )(freq_idces)  # shape: (Nrad, Nphi, Nfreq, Npol)

    # Reorder to (Nrad, Nphi, Nfreq, Npol)
    return jnp.transpose(interp_vals, (3,2,0,1))

def fourier_sum_single(phi_k, a_k, b_k):
    """
    phi_k: (Ncoeff,)
    a_k, b_k: (Ncoeff, ...) where ... is any shape (Nfreq, Npol or empty)
    """
    # Reshape phi_k to broadcast against trailing axes
    # phi_k -> (Ncoeff, 1, 1, ...)
    phi_k_b = phi_k.reshape(phi_k.shape[0], *([1] * (a_k.ndim - 1)))

    # Compute the sum over the coefficient axis
    return jnp.sum(
        a_k * jnp.cos(phi_k_b) +
        b_k * jnp.sin(phi_k_b),
        axis=0
    )

batched_fourier_sum = jax.vmap(
            jax.vmap(fourier_sum_single,
                     in_axes=(0, 0, 0)),     # map over Ny
            in_axes=(0, 0, 0)                # map over Nx
        )

batched_fourier_sum_1d = jax.vmap(fourier_sum_single, in_axes=(0, 0, 0))     # map over Nant