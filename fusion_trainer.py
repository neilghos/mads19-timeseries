import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc
import h5py

from internal_models import MobileNetV1

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FusionH5Dataset(Dataset):
    def __init__(self, wave_h5_path, mads_h5_path):
        self.wave_h5_path = wave_h5_path
        self.mads_h5_path = mads_h5_path
        
        with h5py.File(self.wave_h5_path, 'r') as hf:
            self.length = len(hf['target'])
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Open per-worker to avoid h5py multiprocessing issues
        with h5py.File(self.wave_h5_path, 'r') as hf_w, h5py.File(self.mads_h5_path, 'r') as hf_m:
            waveform = hf_w['waveform'][idx]
            mads = hf_m['mads_features'][idx]
            target = hf_w['target'][idx]
        
        # Convert int16 to float32 between -1 and 1
        waveform = waveform.astype(np.float32) / 32768.0
        return (
            torch.tensor(waveform), 
            torch.tensor(mads, dtype=torch.float32), 
            torch.tensor(target, dtype=torch.long)
        )

class NormalizedMLPMixerFusionMobileNetV1(nn.Module):
    def __init__(self, mads_dim=19, classes_num=50, hidden_dim=512):
        super(NormalizedMLPMixerFusionMobileNetV1, self).__init__()
        
        # Base PANNs model
        self.backbone = MobileNetV1(
            sample_rate=32000, window_size=1024, hop_size=320, 
            mel_bins=64, fmin=50, fmax=14000, classes_num=527
        )
        
        # Remove the original AudioSet classifier
        self.backbone.fc_audioset = nn.Identity()
        
        # Normalize the raw MADS physical priors
        self.mads_norm = nn.BatchNorm1d(mads_dim)
        
        # MLP Mixer: Linear -> BatchNorm -> GELU -> Linear (Classifier)
        # 1024 (MobileNetV1 emb_dim) + 19 (MADS)
        self.mixer = nn.Sequential(
            nn.Linear(1024 + mads_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes_num)
        )
        
    def forward(self, waveform, mads_features):
        # Neural Embedding
        output_dict = self.backbone(waveform)
        embedding = output_dict['embedding']
        
        # Normalize MADS
        mads_normed = self.mads_norm(mads_features)
        
        # Concatenate Deep Embedding and Normalized Physical Prior
        fused_vector = torch.cat([embedding, mads_normed], dim=1)
        
        # Non-linear Mixer to Classification
        logits = self.mixer(fused_vector)
        return logits

def train_and_evaluate(epochs=50, batch_size=32, lr=1e-4):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_accuracies = []
    
    h5_dir = os.path.join("resources", "hdf5s", "esc50")
    
    for fold in range(1, 6):
        print(f"\n--- Normalized MLP Mixer Fusion Fold {fold} ---")
        
        train_w_path = os.path.join(h5_dir, f"train_fold{fold}.h5")
        train_m_path = os.path.join(h5_dir, f"mads_train_fold{fold}.h5")
        test_w_path = os.path.join(h5_dir, f"test_fold{fold}.h5")
        test_m_path = os.path.join(h5_dir, f"mads_test_fold{fold}.h5")
        
        train_dataset = FusionH5Dataset(train_w_path, train_m_path)
        test_dataset = FusionH5Dataset(test_w_path, test_m_path)
        
        # num_workers=4 for faster HDF5 loading
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Load Model
        model = NormalizedMLPMixerFusionMobileNetV1(mads_dim=19, classes_num=50)
        
        # Load Pretrained Weights into backbone
        weight_path = "weights/MobileNetV1.pth"
        if not os.path.exists(weight_path):
            raise FileNotFoundError("Missing weights for MobileNetV1.")
            
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # Filter out fc_audioset weights since we replaced it with Identity
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc_audioset.")}
        
        # Load into backbone (strict=False because fc_audioset is missing from state_dict)
        model.backbone.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for waveforms, mads, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                waveforms, mads, labels = waveforms.to(device), mads.to(device), labels.to(device)
                optimizer.zero_grad()
                
                logits = model(waveforms, mads)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for waveforms, mads, labels in test_loader:
                    waveforms, mads, labels = waveforms.to(device), mads.to(device), labels.to(device)
                    logits = model(waveforms, mads)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
            acc = (correct / total) * 100
            if acc > best_acc:
                best_acc = acc
                
        print(f"Fold {fold} Best Accuracy: {best_acc:.2f}%")
        fold_accuracies.append(best_acc)
        
        # Memory cleanup
        del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    return mean_acc, std_acc

def main():
    print("Starting Normalized MLP Mixer Fusion with MobileNetV1...")
    mean_acc, std_acc = train_and_evaluate()
    result_str = f"{mean_acc:.2f} ± {std_acc:.2f}%"
    print(f"--- MobileNetV1 Normalized MLP Mixer Fusion Final: {result_str} ---")
    
    # Save results
    results = {"Model": "MobileNetV1_MLPMixerFusion", "Accuracy": result_str}
    df = pd.DataFrame([results])
    df.to_csv("fusion_results_esc50.csv", mode='a', header=not os.path.exists("fusion_results_esc50.csv"), index=False)
    print("Saved fusion baselines to fusion_results_esc50.csv")

if __name__ == "__main__":
    main()
