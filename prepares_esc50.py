import os
import csv
import argparse
import logging
import wave

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
    log_path = os.path.join(log_dir, "prepare_esc50.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def load_wav_mono_resample(path, target_sr):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        src_sr = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sample_width == 1:
        arr = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        arr = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("Unsupported sample width {} in {}".format(sample_width, path))

    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)

    if src_sr != target_sr:
        src_idx = np.arange(arr.shape[0], dtype=np.float32)
        new_len = int(round(arr.shape[0] * float(target_sr) / float(src_sr)))
        dst_idx = np.linspace(0, arr.shape[0] - 1, num=max(new_len, 1), dtype=np.float32)
        arr = np.interp(dst_idx, src_idx, arr).astype(np.float32)

    return arr


def pad_or_truncate(x, audio_length):
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=x.dtype)), axis=0)
    return x[0:audio_length]


def float32_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


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


def write_label_csv(rows, out_path):
    create_folder(os.path.dirname(out_path))
    pairs = sorted({(int(r["target"]), r["category"]) for r in rows}, key=lambda x: x[0])
    with open(out_path, "w", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["index", "label"])
        for idx, label in pairs:
            writer.writerow([idx, label])


def write_fold_hdf5(rows, audio_dir, out_path, sample_rate, clip_samples):
    create_folder(os.path.dirname(out_path))
    n = len(rows)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("audio_name", shape=(n,), dtype="S128")
        hf.create_dataset("waveform", shape=(n, clip_samples), dtype=np.int16)
        hf.create_dataset("target", shape=(n,), dtype=np.int64)
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        for i, row in enumerate(rows):
            name = row["filename"]
            target = int(row["target"])
            path = os.path.join(audio_dir, name)
            waveform = load_wav_mono_resample(path, sample_rate)
            waveform = pad_or_truncate(waveform, clip_samples)
            hf["audio_name"][i] = name.encode()
            hf["waveform"][i] = float32_to_int16(waveform)
            hf["target"][i] = target


def prepare(args):
    sample_rate = args.sample_rate
    clip_samples = int(sample_rate * args.clip_duration)
    csv_path = resolve_meta_path(args.dataset_dir)
    audio_dir = find_audio_dir(args.dataset_dir)
    out_dir = resource_path("hdf5s", "esc50")

    create_logging(resource_path("logs", "prepare_esc50"))
    rows = load_esc50_rows(csv_path)
    write_label_csv(rows, resource_path("metadata", "esc50_class_labels_indices.csv"))

    for fold in [1, 2, 3, 4, 5]:
        train_rows = [r for r in rows if int(r["fold"]) != fold]
        test_rows = [r for r in rows if int(r["fold"]) == fold]
        write_fold_hdf5(train_rows, audio_dir, os.path.join(out_dir, "train_fold{}.h5".format(fold)), sample_rate, clip_samples)
        write_fold_hdf5(test_rows, audio_dir, os.path.join(out_dir, "test_fold{}.h5".format(fold)), sample_rate, clip_samples)
    logging.info("ESC-50 preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_duration", type=float, default=5.0)
    args = parser.parse_args()
    prepare(args)
