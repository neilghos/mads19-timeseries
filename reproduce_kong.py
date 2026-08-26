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

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['target'])
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Open per-worker to avoid h5py multiprocessing issues
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = hf['waveform'][idx]
            target = hf['target'][idx]
        
        # Convert int16 to float32 between -1 and 1
        waveform = waveform.astype(np.float32) / 32768.0
        return torch.tensor(waveform), torch.tensor(target, dtype=torch.long)

def get_model_and_dim(model_name):
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

def train_and_evaluate(model_name, epochs=50, batch_size=32, lr=1e-4):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_accuracies = []
    
    h5_dir = os.path.join("resources", "hdf5s", "esc50")
    
    for fold in range(1, 6):
        print(f"\n--- {model_name} Fold {fold} ---")
        
        train_path = os.path.join(h5_dir, f"train_fold{fold}.h5")
        test_path = os.path.join(h5_dir, f"test_fold{fold}.h5")
        
        train_dataset = H5Dataset(train_path)
        test_dataset = H5Dataset(test_path)
        
        # num_workers=4 for faster HDF5 loading
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Load Pretrained Model
        model, emb_dim = get_model_and_dim(model_name)
        weight_path = f"weights/{model_name}.pth"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weights for {model_name}. Did the download finish?")
            
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        # Transfer Learning: Replace final head for 50 classes
        model.fc_audioset = nn.Linear(emb_dim, 50)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train Heads
        model.train()
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                waveforms, labels = waveforms.to(device), labels.to(device)
                optimizer.zero_grad()
                
                output = model(waveforms)
                loss = criterion(output['clipwise_output'], labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Evaluate at the end of each epoch to save the best val accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for waveforms, labels in test_loader:
                    waveforms, labels = waveforms.to(device), labels.to(device)
                    output = model(waveforms)
                    preds = output['clipwise_output'].argmax(dim=1)
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
    models_to_test = ["Cnn6", "Cnn10", "MobileNetV1", "MobileNetV2"]
    results = {}
    
    for model_name in models_to_test:
        mean_acc, std_acc = train_and_evaluate(model_name)
        results[model_name] = f"{mean_acc:.2f} ± {std_acc:.2f}%"
        print(f"--- {model_name} Final: {results[model_name]} ---")
        
    df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    df.to_csv("kong_baselines_esc50.csv", index=False)
    print("Saved baselines to kong_baselines_esc50.csv")

if __name__ == "__main__":
    main()
