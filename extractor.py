import numpy as np
import mads_ts

def extract_mads_for_dataset(X, fs=32000):
    """
    Extracts MADS19 features for a list of 1D numpy arrays.
    Args:
        X: List of 1D numpy arrays (waveforms).
        fs: Sampling frequency.
    Returns:
        np.ndarray of shape (len(X), 19)
    """
    features = []
    for x in X:
        feat = mads_ts.extract_mads_ts(x, fs=fs)
        features.append(feat)
    return np.array(features, dtype=np.float32)
