import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm

import esc50loader
import extractor
import baseline
import classifier

CACHE_DIR = "/home/utsab/Data/mads/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_or_extract(name, dataset, extract_func, *args, **kwargs):
    cache_path = os.path.join(CACHE_DIR, f"{dataset}_{name}.npy")
    if os.path.exists(cache_path):
        print(f"Loading cached {name} features from {cache_path}...")
        return np.load(cache_path)
    print(f"Extracting {name} features...")
    t0 = time.time()
    feats = extract_func(*args, **kwargs)
    t1 = time.time()
    print(f"Extraction took {t1-t0:.2f} seconds.")
    np.save(cache_path, feats)
    return feats

def run_cross_validation(X, y, folds, num_classes):
    print(f"Running 5-fold CV for feature shape {X.shape}")
    results = {
        "Linear_Reg": [], "Random_Forest": [], "XGBoost": [], "KMeans": [],
        "MLP_Probe": [], "CNN_Probe": [], "BiGRU_Probe": []
    }
    
    for f in range(1, 6):
        print(f"  Fold {f}...")
        train_idx = np.where(folds != f)[0]
        test_idx = np.where(folds == f)[0]
        
        # Scale features
        scaler = StandardScaler()
        # For 3D features (N, T, C), we need to reshape
        if len(X.shape) == 3:
            B_tr, T_tr, C_tr = X[train_idx].shape
            X_tr = scaler.fit_transform(X[train_idx].reshape(-1, C_tr)).reshape(B_tr, T_tr, C_tr)
            B_te, T_te, C_te = X[test_idx].shape
            X_te = scaler.transform(X[test_idx].reshape(-1, C_te)).reshape(B_te, T_te, C_te)
        else:
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            
        y_tr = y[train_idx]
        y_te = y[test_idx]
        
        fold_res = classifier.evaluate_all(X_tr, y_tr, X_te, y_te, num_classes)
        for k, v in fold_res.items():
            results[k].append(v)
            
    final_res = {}
    for k, v in results.items():
        mean_acc = np.mean(v) * 100
        std_acc = np.std(v) * 100
        final_res[k] = f"{mean_acc:.2f} ± {std_acc:.2f}%"
        print(f"    {k}: {final_res[k]}")
    return final_res

def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Time Series Benchmark")
    parser.add_argument("--dataset", type=str, default="esc50", help="Dataset name (e.g., esc50, FordA)")
    parser.add_argument("--all", action="store_true", help="Run baselines + MADS")
    parser.add_argument("--mads", action="store_true", help="Run MADS only")
    args = parser.parse_args()

    # Load Data
    if args.dataset.lower() == "esc50":
        loader = esc50loader.ESC50Loader()
        print("Loading ESC-50 dataset...")
        X, y, folds = loader.load_all_data()
        num_classes = 50
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented yet.")
        
    print(f"Loaded {len(X)} samples.")
    
    experiment_results = {}

    if args.mads or args.all:
        print("\n--- Evaluating MADS19 ---")
        X_mads = get_cached_or_extract("mads19", args.dataset, extractor.extract_mads_for_dataset, X)
        experiment_results["MADS19 (19D)"] = run_cross_validation(X_mads, y, folds, num_classes)

    if args.all:
        print("\n--- Evaluating catch22 ---")
        X_c22 = get_cached_or_extract("catch22", args.dataset, baseline.extract_catch22, X)
        experiment_results["catch22 (22D)"] = run_cross_validation(X_c22, y, folds, num_classes)
        
        print("\n--- Evaluating TSFEL ---")
        X_tsfel = get_cached_or_extract("tsfel", args.dataset, baseline.extract_tsfel, X)
        experiment_results["TSFEL"] = run_cross_validation(X_tsfel, y, folds, num_classes)
        
        print("\n--- Evaluating TSFRESH ---")
        X_tsfresh = get_cached_or_extract("tsfresh", args.dataset, baseline.extract_tsfresh, X)
        experiment_results["TSFRESH"] = run_cross_validation(X_tsfresh, y, folds, num_classes)
        
        print("\n--- Evaluating B1 & B2 ---")
        cache_b1 = os.path.join(CACHE_DIR, f"{args.dataset}_B1.npy")
        cache_b2 = os.path.join(CACHE_DIR, f"{args.dataset}_B2.npy")
        if os.path.exists(cache_b1) and os.path.exists(cache_b2):
            X_b1 = np.load(cache_b1)
            X_b2 = np.load(cache_b2)
        else:
            X_b1, X_b2 = baseline.extract_b1_b2(X)
            np.save(cache_b1, X_b1)
            np.save(cache_b2, X_b2)
            
        print("B1 (26D)")
        experiment_results["B1 (26D)"] = run_cross_validation(X_b1, y, folds, num_classes)
        print("B2 (38D)")
        experiment_results["B2 (38D)"] = run_cross_validation(X_b2, y, folds, num_classes)
        
        # MiniRocket, BOSS, WEASEL require train/test split during fit_transform
        # For simplicity in 5-fold CV, we can do it inside the fold loop, or cache full 
        # But random kernels need to fit on training data.
        # We will add them later or do a special cross_validation loop for them.
        print("\nNote: MiniRocket, BOSS, WEASEL are deferred as they require fitting on train splits directly.")

    # Summary
    print("\n=======================================================")
    print(f"   FINAL BENCHMARK SUMMARY FOR {args.dataset.upper()}")
    print("=======================================================")
    
    df_res = pd.DataFrame(experiment_results).T
    print(df_res.to_markdown())

if __name__ == "__main__":
    main()
