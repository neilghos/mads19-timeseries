import numpy as np
# Compatibility aliases for older librosa on newer NumPy versions.
if not hasattr(np, "complex"):
    np.complex = np.complex128
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import librosa
from pathlib import Path
import warnings

from scipy.signal import find_peaks
from scipy.stats import kurtosis


FEATURE_NAMES = [
    "spectral_anchor_i",
    "spectral_anchor_ii",
    "spectral_anchor_iii",
    "attack_gradient",
    "peak_mode",
    "entropy_flow",
    "kinetic_energy_entropy",
    "energy_phase_correlation",
    "mechanical_efficiency",
    "damping_coefficient",
    "temporal_jitter",
    "resonant_spread",
    "dissipation_limit",
    "pde_residual",
    "zcr_variability",
    "periodicity_strength",
    "pulse_density",
    "impact_peakedness",
    "mechanical_instability_index",
]


def load_audio(audio_file, sr_target=32000, duration_sec=5):
    audio, sr = librosa.load(str(audio_file), sr=sr_target, mono=True)
    audio = librosa.util.normalize(audio)
    audio = librosa.util.fix_length(audio, size=sr_target * duration_sec)
    return audio, sr


def extract_features(audio_file):
    # Keep the training API stable while making the 19-D path self-contained.
    return mel_image_224(audio_file)


def mel_image_224(audio_file):
    audio, sr = load_audio(audio_file)
    return extract_from_array(audio, sr)

def extract_from_array(audio, sr=32000):
    seq_len = len(audio)
    
    # Dynamically scale the FFT window size to handle short general time-series (e.g. len 96)
    # without destroying the structural data through massive zero-padding.
    n_fft = min(2048, seq_len)
    # Ensure hop length is valid and at least 1
    hop_length = max(1, n_fft // 4)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=3, n_fft=n_fft, hop_length=hop_length)
    mfcc_anchors = np.mean(mfcc, axis=1)

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    entropy_flow = np.mean(flatness)

    zcr_series = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_std = np.std(zcr_series)

    ac = librosa.autocorrelate(audio, max_size=2000)
    ac = ac / (ac[0] + 1e-10)
    periodicity_strength = float(np.max(ac[50:])) if len(ac) > 50 else 0.0

    v = np.diff(audio)
    a = np.diff(np.diff(audio))
    centroid_series = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    m_eff = (np.mean(centroid_series) + np.std(centroid_series)) / sr

    ke_series = 0.5 * m_eff * (v**2)
    ke_norm = (ke_series + 1e-10) / np.sum(ke_series + 1e-10)
    ke_entropy = -np.sum(ke_norm * np.log2(ke_norm + 1e-10))

    work = (m_eff * a) * v[:-1]
    mech_efficiency = np.mean(ke_series) / (np.abs(np.mean(work)) + 1e-10)
    mech_efficiency = np.log1p(np.abs(mech_efficiency))

    work_std = np.std(work)
    jerk_var = np.var(np.diff(a)) if len(a) > 2 else 0.0
    momentum = m_eff * np.mean(np.abs(v))
    psi_ratio = (jerk_var * work_std) / (momentum + 1e-10)
    psi_ratio = np.log1p(np.abs(psi_ratio))

    pe = 0.5 * m_eff * (audio**2)
    try:
        c = np.corrcoef(ke_series[: len(pe)], pe[: len(ke_series)])
        energy_swap = float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0
    except Exception:
        energy_swap = 0.0

    psd = np.abs(np.fft.rfft(audio)) ** 2 / len(audio)
    fft_freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    peak_mode = float(fft_freqs[np.argmax(psd)])
    peak_mode = np.log1p(np.abs(peak_mode))

    peaks, _ = find_peaks(psd, distance=50, prominence=np.max(psd) * 0.01)
    combined_hessian = 0.0
    if len(peaks) > 0:
        for p in peaks[np.argsort(psd[peaks])][-10:]:
            if 0 < p < len(psd) - 1:
                combined_hessian += np.abs(psd[p + 1] - 2 * psd[p] + psd[p - 1])

    accel_t = np.diff(audio, n=2)
    pde_consistency = np.abs(np.mean(accel_t**2) - (combined_hessian / (sr**2)))
    pde_residual = pde_consistency / (np.var(audio) + 1e-10)

    # Use dynamic hop_length instead of hardcoded 512
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft)
    damping_coeff = np.std(onset) / (np.mean(onset) + 1e-10)

    envelope = np.abs(onset)
    start_candidates = np.where(envelope > np.max(envelope) * 0.05)[0]
    if len(start_candidates) == 0:
        attack_gradient = 0.0
    else:
        start_idx = int(start_candidates[0])
        end_idx = min(len(envelope), start_idx + int(0.05 * sr / hop_length))
        attack_segment = envelope[start_idx:end_idx]
        attack_gradient = float(np.mean(np.gradient(attack_segment))) if len(attack_segment) > 1 else 0.0

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    col_sum = np.sum(S, axis=0) + 1e-10
    centroid_frames = np.sum(freqs[:, None] * S, axis=0) / col_sum
    spread = np.mean(np.sqrt(np.sum(((freqs[:, None] - centroid_frames[None, :]) ** 2) * S, axis=0) / col_sum))

    pulse_density = len(find_peaks(np.abs(audio), height=np.max(np.abs(audio)) * 0.3)[0]) / (len(audio) / sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85, n_fft=n_fft, hop_length=hop_length)[0]
    friction_point = np.mean(rolloff)

    flux_gradient = np.std(np.diff(onset)) if len(onset) > 1 else 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation due to catastrophic cancellation.*",
            category=RuntimeWarning,
        )
        spec_k = kurtosis(S, axis=0, fisher=True, nan_policy="omit")
    spec_k = np.nan_to_num(spec_k, nan=0.0, posinf=0.0, neginf=0.0)
    impact_peakedness = float(np.mean(spec_k))

    features = np.concatenate(
        [
            mfcc_anchors,
            [attack_gradient],
            [peak_mode],
            [entropy_flow],
            [ke_entropy],
            [energy_swap],
            [mech_efficiency],
            [damping_coeff],
            [flux_gradient],
            [spread],
            [friction_point],
            [pde_residual],
            [zcr_std],
            [periodicity_strength],
            [pulse_density],
            [impact_peakedness],
            [psi_ratio],
        ]
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -1e6, 1e6)
    return features.astype(np.float32)


def extract_features_from_path(audio_file):
    return extract_features(Path(audio_file))
