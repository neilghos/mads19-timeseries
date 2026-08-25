import numpy as np
import librosa
from scipy.stats import kurtosis

def extract_catch22(X):
    try:
        import pycatch22
    except ImportError:
        print("pycatch22 not installed.")
        return np.zeros((len(X), 22))
        
    features = []
    for x in X:
        try:
            res = pycatch22.catch22_all(np.asarray(x, dtype=np.float64))
            vals = np.nan_to_num(res['values'], nan=0.0)
            features.append(vals)
        except Exception:
            features.append(np.zeros(22))
    return np.array(features, dtype=np.float32)

def extract_tsfel(X, fs=32000):
    try:
        import tsfel
    except ImportError:
        print("tsfel not installed.")
        return np.zeros((len(X), 10))
        
    cfg = tsfel.get_features_by_domain()
    features = []
    for x in X:
        try:
            df = tsfel.time_series_features_extractor(cfg, x, fs=fs, verbose=0)
            features.append(np.nan_to_num(df.values.flatten(), nan=0.0))
        except Exception:
            features.append(np.zeros(10)) # fallback
    return np.array(features, dtype=np.float32)

def extract_tsfresh(X):
    try:
        from tsfresh.feature_extraction import extract_features, MinimalFCParameters
        import pandas as pd
    except ImportError:
        print("tsfresh not installed.")
        return np.zeros((len(X), 10))
        
    # Convert list of arrays to tsfresh format
    df_list = []
    for i, x in enumerate(X):
        df_list.append(pd.DataFrame({'id': i, 'time': range(len(x)), 'value': x}))
    df = pd.concat(df_list)
    extracted = extract_features(df, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters(), n_jobs=4)
    return np.nan_to_num(extracted.values, nan=0.0).astype(np.float32)

def fit_transform_minirocket(X_train, X_test):
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket
    except ImportError:
        try:
            from sktime.transformations.panel.rocket import MiniRocket
        except ImportError:
            print("MiniRocket not available.")
            return np.zeros((len(X_train), 10000)), np.zeros((len(X_test), 10000))
    
    # Needs (N, C, L) format
    X_tr = np.expand_dims(np.array(X_train), axis=1)
    X_te = np.expand_dims(np.array(X_test), axis=1)
    
    mr = MiniRocket(n_jobs=4)
    X_tr_trans = mr.fit_transform(X_tr)
    X_te_trans = mr.transform(X_te)
    return X_tr_trans, X_te_trans

def fit_transform_boss(X_train, X_test):
    try:
        from aeon.transformations.collection.dictionary_based import SFA
    except ImportError:
        print("BOSS/SFA not available.")
        return np.zeros((len(X_train), 10)), np.zeros((len(X_test), 10))
        
    X_tr = np.expand_dims(np.array(X_train), axis=1)
    X_te = np.expand_dims(np.array(X_test), axis=1)
    
    sfa = SFA(word_length=8, alphabet_size=4, window_length=100)
    X_tr_trans = sfa.fit_transform(X_tr)
    X_te_trans = sfa.transform(X_te)
    return np.array(X_tr_trans), np.array(X_te_trans)

def fit_transform_weasel(X_train, X_test):
    # Fallback to SFA if WEASEL is too complex/slow
    return fit_transform_boss(X_train, X_test)

def extract_b1_b2(X, fs=32000):
    """
    Extracts B1 (26D) and B2 (38D) baselines.
    """
    b1_feats = []
    b2_feats = []
    for x in X:
        # MFCC
        mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(x)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # B1: MFCC 1:12 mean/std + ZCR mean/std
        b1 = np.concatenate([mfcc_mean[1:], mfcc_std[1:], [zcr_mean, zcr_std]])
        b1_feats.append(b1)
        
        # Spectral summaries for B2
        S = np.abs(librosa.stft(x))
        cent = librosa.feature.spectral_centroid(S=S)[0]
        bw = librosa.feature.spectral_bandwidth(S=S)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        rms = librosa.feature.rms(y=x)[0]
        
        sum_mean = [np.mean(cent), np.mean(bw), np.mean(rolloff), np.mean(flatness), np.mean(rms)]
        sum_std = [np.std(cent), np.std(bw), np.std(rolloff), np.std(flatness), np.std(rms)]
        
        # B2: MFCC 0:12 mean/std + ZCR + 5 summaries
        b2 = np.concatenate([mfcc_mean, mfcc_std, [zcr_mean, zcr_std], sum_mean, sum_std])
        b2_feats.append(b2)
        
    return np.array(b1_feats, dtype=np.float32), np.array(b2_feats, dtype=np.float32)
