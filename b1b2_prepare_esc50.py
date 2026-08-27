import os
import csv
import argparse
import logging
import warnings
import librosa

import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    return os.path.join(REPO_ROOT, "resources", *parts)


def resolve_meta_path(dataset_dir):
    candidates = [
        os.path.join(dataset_dir, "meta", "esc50.csv"),
        os.path.join(dataset_dir, "esc50.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Could not find esc50.csv under {}".format(dataset_dir))


def create_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def create_logging(log_dir):
    create_folder(log_dir)
    log_path = os.path.join(log_dir, "b1b2_prepare_esc50.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def find_audio_dir(dataset_dir):
    candidates = [os.path.join(dataset_dir, "audio", "audio"), os.path.join(dataset_dir, "audio")]
    for path in candidates:
        if os.path.isdir(path):
            wavs = [name for name in os.listdir(path) if name.lower().endswith(".wav")]
            if wavs:
                return path
    for root, _, files in os.walk(dataset_dir):
        if any(name.lower().endswith(".wav") for name in files):
            return root
    raise FileNotFoundError("Could not find audio directory under {}".format(dataset_dir))


def load_esc50_rows(csv_path):
    with open(csv_path, "r", newline="") as fr:
        rows = list(csv.DictReader(fr))
    if len(rows) == 0:
        raise RuntimeError("No rows in {}".format(csv_path))
    return rows


def extract_b1_b2(audio_path, sr=32000):
    """
    B1 (26D): Mean and Variance of 13 MFCCs.
    B2 (38D): Mean and Variance of (13 MFCCs + Centroid + Bandwidth + Rolloff + ZCR + RMS + Flatness).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        y = librosa.util.normalize(y)
        
        # Extract features over time
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        rms = librosa.feature.rms(y=y)
        flatness = librosa.feature.spectral_flatness(y=y)
        
        # B1: 26D (MFCC mean + var)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        b1_features = np.concatenate([mfcc_mean, mfcc_var])
        
        # B2: 38D (19 features mean + var)
        extra_features = np.vstack([centroid, bandwidth, rolloff, zcr, rms, flatness])
        extra_mean = np.mean(extra_features, axis=1)
        extra_var = np.var(extra_features, axis=1)
        b2_features = np.concatenate([mfcc_mean, extra_mean, mfcc_var, extra_var])
        
    return b1_features.astype(np.float32), b2_features.astype(np.float32)


def write_fold_hdf5(rows, audio_dir, out_b1_path, out_b2_path, sample_rate):
    create_folder(os.path.dirname(out_b1_path))
    n = len(rows)
    with h5py.File(out_b1_path, "w") as hf_b1, h5py.File(out_b2_path, "w") as hf_b2:
        hf_b1.create_dataset("audio_name", shape=(n,), dtype="S128")
        hf_b1.create_dataset("b1_features", shape=(n, 26), dtype=np.float32)
        hf_b1.create_dataset("target", shape=(n,), dtype=np.int64)
        hf_b1.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)
        
        hf_b2.create_dataset("audio_name", shape=(n,), dtype="S128")
        hf_b2.create_dataset("b2_features", shape=(n, 38), dtype=np.float32)
        hf_b2.create_dataset("target", shape=(n,), dtype=np.int64)
        hf_b2.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        for i, row in enumerate(rows):
            name = row["filename"]
            target = int(row["target"])
            path = os.path.join(audio_dir, name)
            
            b1, b2 = extract_b1_b2(path, sr=sample_rate)
            
            hf_b1["audio_name"][i] = name.encode()
            hf_b1["b1_features"][i] = b1
            hf_b1["target"][i] = target
            
            hf_b2["audio_name"][i] = name.encode()
            hf_b2["b2_features"][i] = b2
            hf_b2["target"][i] = target
            
            if i % 100 == 0:
                logging.info(f"Processed {i}/{n} files...")


def prepare(args):
    sample_rate = args.sample_rate
    csv_path = resolve_meta_path(args.dataset_dir)
    audio_dir = find_audio_dir(args.dataset_dir)
    out_dir = resource_path("hdf5s", "esc50")

    create_logging(resource_path("logs", "prepare_esc50"))
    rows = load_esc50_rows(csv_path)

    for fold in [1, 2, 3, 4, 5]:
        logging.info(f"Extracting B1 (26D) and B2 (38D) features for Fold {fold}...")
        train_rows = [r for r in rows if int(r["fold"]) != fold]
        test_rows = [r for r in rows if int(r["fold"]) == fold]
        
        write_fold_hdf5(
            train_rows, audio_dir, 
            os.path.join(out_dir, "b1_train_fold{}.h5".format(fold)),
            os.path.join(out_dir, "b2_train_fold{}.h5".format(fold)),
            sample_rate
        )
        write_fold_hdf5(
            test_rows, audio_dir, 
            os.path.join(out_dir, "b1_test_fold{}.h5".format(fold)),
            os.path.join(out_dir, "b2_test_fold{}.h5".format(fold)),
            sample_rate
        )
        
    logging.info("B1 and B2 preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=32000)
    args = parser.parse_args()
    prepare(args)
