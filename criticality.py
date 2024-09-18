import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

def hilbert_phase(data):
    """
    Compute the phase of a time series using Hilbert Transform.
    
    Parameters:
    data: 2D numpy array, shape (nodes, time_points)
    
    Returns:
    phase_data: 2D numpy array, shape (nodes, time_points), phase of the input data
    """
    analytic_signal = hilbert(data, axis=1)
    phase_data      = np.unwrap(np.angle(analytic_signal), axis=1)
    return phase_data

def phase_diff(phase_data):
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

def _normalize_signal(x):
    eps    = 1e-10
    x_abs  = np.abs(x)
    x_norm = x / (x_abs + eps)
    return x_norm

def _dfa_boxcar(data_orig, win_lengths):
    """            
    Computes DFA using FFT-based method. 
    Input: 
        data_orig:   1D array of amplitude time series.
        win_lengths: 1D array of window lengths in samples.
    Output:
        fluct: Fluctuation function.
        slope: Slopes.
    """
    data    = np.array(data_orig, copy=True)
    win_arr = np.array(win_lengths)
    
    data     -= data.mean(axis=1, keepdims=True)
    data_fft = np.fft.fft(data)

    n_chans, n_ts = data.shape
    is_odd        = n_ts % 2 == 1

    nx         = (n_ts + 1) // 2 if is_odd else n_ts // 2 + 1
    data_power = 2 * np.abs(data_fft[:, 1:nx])**2

    if not is_odd:
        data_power[:, -1] /= 2
        
    ff    = np.arange(1, nx)
    g_sin = np.sin(np.pi * ff / n_ts)
    
    hsin = np.sin(np.pi * np.outer(win_arr, ff) / n_ts)
    hcos = np.cos(np.pi * np.outer(win_arr, ff) / n_ts)

    hx = 1 - hsin / np.outer(win_arr, g_sin)
    h  = (hx / (2 * g_sin.reshape(1, -1)))**2

    f2 = np.inner(data_power, h)

    fluct = np.sqrt(f2) / n_ts

    hy = -hx * (hcos * np.pi * ff / n_ts - hsin / win_arr.reshape(-1, 1)) / np.outer(win_arr, g_sin)
    h3 = hy / (4 * g_sin**2)

    slope = np.inner(data_power, h3) / f2 * win_arr
    
    return fluct, slope

def _fit_tukey(x, y, weights):
    """Fit using Tukey's biweight function."""
    N = len(y)
    X = np.array([[1, xi] for xi in x])
    
    rlm_model   = sm.RLM(y, X, M=CustomNorm(weights=weights, c=4.685))
    rlm_results = rlm_model.fit()
    
    b, a = rlm_results.params    
    return a, b

class CustomNorm(sm.robust.norms.TukeyBiweight):
    """Custom Norm Class using Tukey's biweight (bisquare)."""
    
    def __init__(self, weights, c=4.685, **kwargs):
        super().__init__(**kwargs)
        self.weights_vector = np.array(weights)
        self.flag = 0
        self.c = c
        
    def weights(self, z):
        if self.flag == 0:
            self.flag = 1
            return self.weights_vector.copy()
        else:
            subset = self._subset(z)
            return (1 - (z / self.c)**2)**2 * subset

def dfa(data, window_lengths, method='boxcar', fitting='Tukey', weighting='sq1ox'):
    """
    Compute DFA with FFT-based 'boxcar' method.
    
    INPUT:
        data:           2D array of size [N_channels x N_samples]. Should be normalized data!!
        window_lengths: sequence of window sizes, should be in samples.
        method:         'boxcar' method
        fitting:        'Tukey' for biweight/bisquare
        weighting:      'sq1ox' or '1ox' 
                    
    OUTPUT:        
        fluctuation: 2D array of size N_channels x N_windows), 
        slope:       2D array of size N_channels x N_windows), 
        DFA:         1D vector of size N_channels 
        residuals:   1D vector of size N_channels        
    """

    if data.ndim == 1:
        # If data is 1D, convert it to 2D
        data = np.expand_dims(data, axis=0)
        
    N_samp = data.shape[1]
    
    if method == 'boxcar':
        fluct, slope = _dfa_boxcar(data, window_lengths)
        
    if fitting == 'Tukey':
        if weighting == 'sq1ox':
            weights = np.sqrt(np.divide(1, N_samp / window_lengths))
        elif weighting == '1ox':
            weights = np.divide(1, N_samp / window_lengths)
        
        dfa_values = np.zeros(data.shape[0])
        residuals  = np.zeros(data.shape[0])  
        x = np.log2(window_lengths)         
        for i in range(data.shape[0]):
            y = np.log2(fluct[i])
            dfa_values[i], residuals[i] = _fit_tukey(x, y, weights)

    return fluct, slope, dfa_values, residuals


def compute_dfa(phase_diff_matrix, window_lengths):
    """
    Compute DFA exponents for the phase differences.
    
    Parameters:
    phase_diff_matrix: 2D numpy array, shape (pairwise comparisons, time_points)
    window_lengths: list or array of window lengths for DFA
    
    Returns:
    dfa_exponents: 1D numpy array, DFA exponents for each pairwise phase difference
    """
    dfa_exponents = []
    for diff in phase_diff_matrix:
        fluct, slope, exponent, _ = dfa(diff, window_lengths)
        dfa_exponents.append(exponent)
    
    return np.array(dfa_exponents)


