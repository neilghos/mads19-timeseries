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

from internal_models import Cnn6, Cnn10, MobileNetV1, MobileNetV2

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
    def __init__(self, wave_h5_path, feature_h5_path, feature_key):
        self.wave_h5_path = wave_h5_path
        self.feature_h5_path = feature_h5_path
        self.feature_key = feature_key
        
        with h5py.File(self.wave_h5_path, 'r') as hf:
            self.length = len(hf['target'])
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        with h5py.File(self.wave_h5_path, 'r') as hf_w, h5py.File(self.feature_h5_path, 'r') as hf_f:
            waveform = hf_w['waveform'][idx]
            feature_vec = hf_f[self.feature_key][idx]
            target = hf_w['target'][idx]
        
        waveform = waveform.astype(np.float32) / 32768.0
        return (
            torch.tensor(waveform), 
            torch.tensor(feature_vec, dtype=torch.float32), 
            torch.tensor(target, dtype=torch.long)
        )

def get_backbone(model_name):
    if model_name == "Cnn6":
        return Cnn6(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527), 512
    elif model_name == "Cnn10":
        return Cnn10(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527), 512
    elif model_name == "MobileNetV1":
        return MobileNetV1(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527), 1024
    elif model_name == "MobileNetV2":
        return MobileNetV2(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527), 1024
    else:
        raise ValueError(f"Unknown model {model_name}")

class NormalizedMLPMixerFusion(nn.Module):
    def __init__(self, backbone, emb_dim, prior_dim, classes_num=50, hidden_dim=512):
        super(NormalizedMLPMixerFusion, self).__init__()
        
        self.backbone = backbone
        self.backbone.fc_audioset = nn.Identity()
        
        self.prior_norm = nn.BatchNorm1d(prior_dim)
        
        self.mixer = nn.Sequential(
            nn.Linear(emb_dim + prior_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes_num)
        )
        
    def forward(self, waveform, prior_features):
        output_dict = self.backbone(waveform)
        embedding = output_dict['embedding']
        
        prior_normed = self.prior_norm(prior_features)
        fused_vector = torch.cat([embedding, prior_normed], dim=1)
        
        logits = self.mixer(fused_vector)
        return logits

def train_and_evaluate(model_name, feature_type, prior_dim, feature_key, epochs=50, batch_size=32, lr=1e-4):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_accuracies = []
    
    h5_dir = os.path.join("resources", "hdf5s", "esc50")
    
    for fold in range(1, 6):
        print(f"\n--- {model_name} + {feature_type} Fusion Fold {fold} ---")
        
        train_w_path = os.path.join(h5_dir, f"train_fold{fold}.h5")
        train_f_path = os.path.join(h5_dir, f"{feature_type.lower()}_train_fold{fold}.h5")
        test_w_path = os.path.join(h5_dir, f"test_fold{fold}.h5")
        test_f_path = os.path.join(h5_dir, f"{feature_type.lower()}_test_fold{fold}.h5")
        
        train_dataset = FusionH5Dataset(train_w_path, train_f_path, feature_key)
        test_dataset = FusionH5Dataset(test_w_path, test_f_path, feature_key)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        backbone, emb_dim = get_backbone(model_name)
        model = NormalizedMLPMixerFusion(backbone, emb_dim, prior_dim=prior_dim, classes_num=50)
        
        weight_path = f"weights/{model_name}.pth"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weights for {model_name}.")
            
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc_audioset.")}
        
        model.backbone.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for waveforms, priors, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                waveforms, priors, labels = waveforms.to(device), priors.to(device), labels.to(device)
                optimizer.zero_grad()
                
                logits = model(waveforms, priors)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for waveforms, priors, labels in test_loader:
                    waveforms, priors, labels = waveforms.to(device), priors.to(device), labels.to(device)
                    logits = model(waveforms, priors)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
            acc = (correct / total) * 100
            if acc > best_acc:
                best_acc = acc
                
        print(f"Fold {fold} Best Accuracy: {best_acc:.2f}%")
        fold_accuracies.append(best_acc)
        
        del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    return mean_acc, std_acc

def main():
    models_to_test = ["Cnn6", "Cnn10", "MobileNetV1", "MobileNetV2"]
    
    # feature_type: (dimension, h5_dataset_key)
    feature_sets = {
        "B1": (26, "b1_features"),
        "B2": (38, "b2_features"),
        "MADS": (19, "mads_features")
    }
    
    results = []
    
    for feat_name, (prior_dim, feature_key) in feature_sets.items():
        for model_name in models_to_test:
            print(f"\n========================================================")
            print(f"Starting {feat_name} ({prior_dim}D) Fusion with {model_name}...")
            print(f"========================================================\n")
            
            mean_acc, std_acc = train_and_evaluate(
                model_name=model_name, 
                feature_type=feat_name, 
                prior_dim=prior_dim, 
                feature_key=feature_key
            )
            
            result_str = f"{mean_acc:.2f} ± {std_acc:.2f}%"
            print(f"--- {model_name} + {feat_name} Fusion Final: {result_str} ---")
            
            results.append({
                "Model": model_name,
                "Feature_Set": feat_name,
                "Accuracy": result_str
            })
            
            df = pd.DataFrame(results)
            df.to_csv("fusion_ablation_esc50.csv", index=False)
            print("Intermediate results saved to fusion_ablation_esc50.csv")
            
    print("\nAll ablations complete. Final results saved to fusion_ablation_esc50.csv")

if __name__ == "__main__":
    main()
