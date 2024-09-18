import numpy as np

def generate_pink_noise(N, alpha=1):
    """
    Generate pink noise with a 1/f^alpha power spectrum.

    Parameters:
    N: int, Number of time points
    alpha: float, Exponent for the 1/f^alpha power spectrum (alpha=1 gives pink noise)

    Returns:
    pink_noise: 1D numpy array of pink noise
    """
    # Generate white noise
    white_noise = np.random.normal(size=N)
    
    # Fourier transform of white noise
    f    = np.fft.rfftfreq(N)
    f[0] = 1  # To avoid division by zero
    
    # Scale the Fourier coefficients by 1/f^alpha
    spectrum   = np.fft.rfft(white_noise) / (f ** (alpha / 2.0))
    
    # Inverse Fourier transform to obtain pink noise
    pink_noise = np.fft.irfft(spectrum, n=N)
    
    return pink_noise
