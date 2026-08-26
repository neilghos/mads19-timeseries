import os
import csv
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

import mads19

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
    log_path = os.path.join(log_dir, "mads_prepare_esc50.log")
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


def write_fold_hdf5(rows, audio_dir, out_path, sample_rate):
    create_folder(os.path.dirname(out_path))
    n = len(rows)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("audio_name", shape=(n,), dtype="S128")
        hf.create_dataset("mads_features", shape=(n, 19), dtype=np.float32)
        hf.create_dataset("target", shape=(n,), dtype=np.int64)
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        for i, row in enumerate(rows):
            name = row["filename"]
            target = int(row["target"])
            path = os.path.join(audio_dir, name)
            
            # Extract MADS19 Features directly from the audio file
            mads_vector = mads19.extract_features_from_path(path)
            
            hf["audio_name"][i] = name.encode()
            hf["mads_features"][i] = mads_vector
            hf["target"][i] = target
            
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
        logging.info(f"Extracting MADS19 features for Fold {fold}...")
        train_rows = [r for r in rows if int(r["fold"]) != fold]
        test_rows = [r for r in rows if int(r["fold"]) == fold]
        write_fold_hdf5(train_rows, audio_dir, os.path.join(out_dir, "mads_train_fold{}.h5".format(fold)), sample_rate)
        write_fold_hdf5(test_rows, audio_dir, os.path.join(out_dir, "mads_test_fold{}.h5".format(fold)), sample_rate)
        
    logging.info("MADS19 ESC-50 preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=32000)
    args = parser.parse_args()
    prepare(args)
