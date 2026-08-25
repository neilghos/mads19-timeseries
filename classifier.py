import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter

def purity_score(y_true, y_pred):
    """Calculate clustering purity."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total = 0
    for k in np.unique(y_pred):
        idx = np.where(y_pred == k)[0]
        if len(idx) > 0:
            majority_class = Counter(y_true[idx]).most_common(1)[0][0]
            total += np.sum(y_true[idx] == majority_class)
    return total / len(y_true)

def eval_sklearn(X_train, y_train, X_test, y_test, model_name):
    if model_name == "Linear_Reg":
        clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    elif model_name == "Random_Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        clf = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    elif model_name == "KMeans":
        clf = KMeans(n_clusters=len(np.unique(y_train)), random_state=42, n_init='auto')
        clf.fit(X_train)
        preds = clf.predict(X_test)
        return purity_score(y_test, preds)
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")
        
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)

# --- PyTorch Probes ---

class MLPProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.mean(dim=1) # Aggregate if 3D
        return self.net(x)

class CNNProbe(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1) # Add dummy temporal dim if 2D
        else:
            x = x.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        feat = self.conv(x).squeeze(-1)
        return self.fc(feat)

class BiGRUProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.gru = nn.GRU(in_dim, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add dummy temporal dim if 2D
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]) # Last step

def eval_torch_probe(X_train, y_train, X_test, y_test, model_name, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    
    in_dim = X_train.shape[-1]
    
    if model_name == "MLP_Probe":
        model = MLPProbe(in_dim, num_classes)
    elif model_name == "CNN_Probe":
        model = CNNProbe(in_dim, num_classes)
    elif model_name == "BiGRU_Probe":
        model = BiGRUProbe(in_dim, num_classes)
    else:
        raise ValueError(f"Unknown torch probe: {model_name}")
        
    model = model.to(device)
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds = TensorDataset(X_te_t, y_te_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_acc = 0.0
    for epoch in range(100):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                preds = out.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        best_acc = max(best_acc, correct / total)
        
    return best_acc

def evaluate_all(X_train, y_train, X_test, y_test, num_classes=50):
    """Runs all classifiers and returns a dictionary of accuracies."""
    results = {}
    
    # Sklearn models
    for m in ["Linear_Reg", "Random_Forest", "XGBoost", "KMeans"]:
        acc = eval_sklearn(X_train, y_train, X_test, y_test, m)
        results[m] = acc
        
    # Torch probes
    for m in ["MLP_Probe", "CNN_Probe", "BiGRU_Probe"]:
        acc = eval_torch_probe(X_train, y_train, X_test, y_test, m, num_classes)
        results[m] = acc
        
    return results
