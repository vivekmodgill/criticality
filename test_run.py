import numpy as np
from pink_noise import generate_pink_noise
from criticality import *


def test_dfa_with_pink_noise():
    """
    Test the DFA function using generated pink noise.
    """
    N          = 10000  # Number of time points
    pink_noise = generate_pink_noise(N)
    
    # Define window lengths for DFA
    window_lengths = np.logspace(2, 4, num=20, base=2).astype(int)
    
    # Reshape the pink noise to simulate one "channel"
    pink_noise     = pink_noise.reshape(1, -1)  # Shape (1, N)
    
    # Compute DFA
    fluct, slope, dfa_values, residuals = compute_dfa(pink_noise, window_lengths)
    
    print("DFA Exponent for Pink Noise (should be close to 1):", dfa_values)

# Run the test
test_dfa_with_pink_noise()
