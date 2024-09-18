import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

def compute_hilbert_phase(data):
    """
    Compute the phase of a time series using the Hilbert Transform.
    
    Parameters:
    data: 2D numpy array, shape (nodes, time_points)
    
    Returns:
    phase_data: 2D numpy array, shape (nodes, time_points), phase of the input data
    """
    analytic_signal = hilbert(data, axis=1)
    phase_data      = np.unwrap(np.angle(analytic_signal), axis=1)
    return phase_data

def compute_phase_differences(phase_data):
    """
    Compute the pairwise phase differences between all nodes.
    
    Parameters:
    phase_data: 2D numpy array, shape (nodes, time_points)
    
    Returns:
    phase_diff_matrix: 2D numpy array, shape (nodes*(nodes-1)//2, time_points), pairwise phase differences
    """
    num_nodes         = phase_data.shape[0]
    phase_diff_matrix = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            diff = np.abs(phase_data[i] - phase_data[j])
            phase_diff_matrix.append(diff)
    
    return np.array(phase_diff_matrix)

def normalize_signal(signal):
    """
    Normalize a signal to avoid numerical instability.
    
    Parameters:
    signal: 1D or 2D numpy array
    
    Returns:
    normalized_signal: Normalized version of the input signal
    """
    eps = 1e-10
    return signal / (np.abs(signal) + eps)

def _compute_dfa_boxcar(data, win_lengths):
    """
    Compute DFA using an FFT-based 'boxcar' method.
    
    Parameters:
    data: 2D numpy array of amplitude time series (nodes x time_points)
    win_lengths: 1D numpy array of window lengths (in samples)
    
    Returns:
    fluctuation: Fluctuation function
    slope: Slopes of the fluctuation function
    """
    data    = np.array(data, copy=True)
    win_arr = np.array(win_lengths)
    
    data     -= data.mean(axis=1, keepdims=True)
    data_fft = np.fft.fft(data)
    
    n_chans, n_ts = data.shape
    is_odd        = n_ts % 2 == 1
    nx            = (n_ts + 1) // 2 if is_odd else n_ts // 2 + 1
    data_power    = 2 * np.abs(data_fft[:, 1:nx])**2
    
    if not is_odd:
        data_power[:, -1] /= 2
        
    ff    = np.arange(1, nx)
    g_sin = np.sin(np.pi * ff / n_ts)
    
    hsin = np.sin(np.pi * np.outer(win_arr, ff) / n_ts)
    hcos = np.cos(np.pi * np.outer(win_arr, ff) / n_ts)

    hx = 1 - hsin / np.outer(win_arr, g_sin)
    h  = (hx / (2 * g_sin.reshape(1, -1)))**2

    f2          = np.inner(data_power, h)
    fluctuation = np.sqrt(f2) / n_ts
    
    hy    = -hx * (hcos * np.pi * ff / n_ts - hsin / win_arr.reshape(-1, 1)) / np.outer(win_arr, g_sin)
    h3    = hy / (4 * g_sin**2)
    slope = np.inner(data_power, h3) / f2 * win_arr
    
    return fluctuation, slope

def _fit_tukey_regression(x, y, weights):
    """
    Fit a robust linear model using Tukey's biweight function.
    
    Parameters:
    x: 1D numpy array, log-transformed window lengths
    y: 1D numpy array, log-transformed fluctuation values
    weights: Weights for robust fitting
    
    Returns:
    slope: Slope of the fit (DFA exponent)
    intercept: Intercept of the fit
    """
    X = np.column_stack((np.ones_like(x), x))
    
    rlm_model   = sm.RLM(y, X, M=CustomTukeyNorm(weights=weights, c=4.685))
    rlm_results = rlm_model.fit()
    
    intercept, slope = rlm_results.params
    return slope, intercept

class CustomTukeyNorm(sm.robust.norms.TukeyBiweight):
    """
    Custom Norm Class using Tukey's biweight (bisquare).
    
    Parameters:
    weights: Custom weights for robust fitting
    c: Tuning constant (default: 4.685)
    """
    def __init__(self, weights, c=4.685, **kwargs):
        super().__init__(**kwargs)
        self.weights_vector = np.array(weights)
        self.first_pass = True
        self.c = c
        
    def weights(self, z):
        if self.first_pass:
            self.first_pass = False
            return self.weights_vector.copy()
        else:
            subset = self._subset(z)
            return (1 - (z / self.c)**2)**2 * subset

def compute_dfa(data, window_lengths, method='boxcar', fitting='Tukey', weighting='sq1ox'):
    """
    Compute DFA using a specified method and fitting procedure.
    
    Parameters:
    data: 2D array of size [nodes x time_points], normalized data
    window_lengths: 1D array of window lengths (in samples)
    method: DFA computation method (default: 'boxcar')
    fitting: Fitting method for DFA exponent (default: 'Tukey')
    weighting: Weighting method (default: 'sq1ox')
    
    Returns:
    fluctuation: Fluctuation function
    slope: Slopes of the fluctuation function
    dfa_exponents: DFA exponents for each node
    residuals: Residuals of the fit
    """
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
        
    n_samples = data.shape[1]
    
    if method == 'boxcar':
        fluctuation, slope = _compute_dfa_boxcar(data, window_lengths)
        
    if fitting == 'Tukey':
        if weighting == 'sq1ox':
            weights = np.sqrt(1 / (n_samples / window_lengths))
        elif weighting == '1ox':
            weights = 1 / (n_samples / window_lengths)
        
        dfa_exponents = np.zeros(data.shape[0])
        residuals     = np.zeros(data.shape[0])
        x = np.log2(window_lengths)
        for i in range(data.shape[0]):
            y = np.log2(fluctuation[i])
            dfa_exponents[i], residuals[i] = _fit_tukey_regression(x, y, weights)

    return fluctuation, slope, dfa_exponents, residuals

def compute_dfa_for_phase_differences(phase_diff_matrix, window_lengths):
    """
    Compute DFA exponents for the phase differences.
    
    Parameters:
    phase_diff_matrix: 2D numpy array, pairwise phase differences
    window_lengths: list or array of window lengths for DFA
    
    Returns:
    dfa_exponents: 1D numpy array, DFA exponents for each pairwise phase difference
    """
    dfa_exponents = []
    for diff in phase_diff_matrix:
        _, _, exponent, _ = compute_dfa(diff, window_lengths)
        dfa_exponents.append(exponent)
    
    return np.array(dfa_exponents)
