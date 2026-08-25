import os
import pandas as pd
import librosa
import numpy as np

class ESC50Loader:
    def __init__(self, data_dir="/home/utsab/Data/mads/ESC-50"):
        self.data_dir = data_dir
        self.meta_csv = os.path.join(self.data_dir, "meta", "esc50.csv")
        self.audio_dir = os.path.join(self.data_dir, "audio")
        
        if not os.path.exists(self.meta_csv):
            raise FileNotFoundError(f"Metadata file not found at {self.meta_csv}")
            
        self.meta_df = pd.read_csv(self.meta_csv)
        self.sr_target = 32000
        self.duration_sec = 5

    def load_all_data(self):
        """
        Loads all audio files and returns raw waveforms, labels, and folds.
        Returns:
            X: list of 1D numpy arrays (waveforms)
            y: list of integers (targets)
            folds: list of integers (folds 1-5)
        """
        X = []
        y = []
        folds = []
        
        for idx, row in self.meta_df.iterrows():
            filepath = os.path.join(self.audio_dir, row['filename'])
            audio, _ = librosa.load(filepath, sr=self.sr_target, mono=True)
            audio = librosa.util.normalize(audio)
            audio = librosa.util.fix_length(audio, size=self.sr_target * self.duration_sec)
            
            X.append(audio)
            y.append(row['target'])
            folds.append(row['fold'])
            
        return X, np.array(y), np.array(folds)
        
    def get_metadata(self):
        """Returns the pandas dataframe of the metadata."""
        return self.meta_df
