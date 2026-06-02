import os
import argparse
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import DAGCN

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_session_data(mat_path, label_arr, feature_type="LDS"):
    """
    Loads EEG features from a single SEED session .mat file.
    SEED session files contain 15 trials (1-based keys: de_LDS1 to de_LDS15).
    Trials 1-9 are training data; Trials 10-15 are testing data.
    """
    data = sio.loadmat(mat_path)
    
    train_x, train_y = [], []
    test_x, test_y = [], []
    
    for trial_idx in range(1, 16):
        de_key = f"de_{feature_type}{trial_idx}"
        psd_key = f"psd_{feature_type}{trial_idx}"
        
        if de_key not in data or psd_key not in data:
            raise KeyError(f"Keys {de_key} or {psd_key} not found in {mat_path}")
            
        # Original shapes: (62, num_samples, 5)
        de_feat = data[de_key]
        psd_feat = data[psd_key]
        
        # Transpose to (num_samples, 62, 5)
        de_feat = de_feat.transpose(1, 0, 2)
        psd_feat = psd_feat.transpose(1, 0, 2)
        
        # Concatenate DE and PSD features along the frequency band dimension (axis=2)
        # Results in (num_samples, 62, 10)
        trial_feat = np.concatenate([de_feat, psd_feat], axis=-1)
        
        num_samples = trial_feat.shape[0]
        trial_label = label_arr[trial_idx - 1]
        trial_labels = np.full(num_samples, trial_label, dtype=np.int64)
        
        if trial_idx <= 9:
            train_x.append(trial_feat)
            train_y.append(trial_labels)
        else:
            test_x.append(trial_feat)
            test_y.append(trial_labels)
            
    # Concatenate all trials
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    
    return train_x, train_y, test_x, test_y

def train_and_evaluate(mat_path, label_arr, args):
    # 1. Load Data
    train_x_np, train_y_np, test_x_np, test_y_np = load_session_data(
        mat_path, label_arr, feature_type=args.feature_type
    )
    
    # 2. Convert to Torch Tensors
    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.long)
    test_x = torch.tensor(test_x_np, dtype=torch.float32)
    test_y = torch.tensor(test_y_np, dtype=torch.long)
    
    # 3. Z-Score Normalization along sample dimension
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True) + 1e-6
    
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    
    # 4. Create DataLoaders
    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Initialize Model
    model = DAGCN('seed').cuda()
    
    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
    best_acc = 0.0
    
    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Evaluation Loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
            
        if args.verbose and (epoch % 20 == 0 or epoch == args.epochs or epoch == 1):
            print(f"Epoch {epoch:3d}/{args.epochs}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}%, Test Loss = {test_loss:.4f}, Test Acc = {test_acc*100:.2f}%")
            
    return best_acc

def main():
    parser = argparse.ArgumentParser(description="DAGCN on SEED Training Pipeline")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\Dev\BCI\EEG_Dataset\SEED\SEED\SEED_EEG\ExtractedFeatures_1s", help="Path to SEED ExtractedFeatures_1s directory")
    parser.add_argument("--session", type=str, default=None, help="Name of a specific session file (e.g. 1_20131027.mat) to run on")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay regularization")
    parser.add_argument("--feature_type", type=str, default="LDS", choices=["LDS", "movingAve", "raw"], help="Feature smoothing type")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type")
    parser.add_argument("--run_all", action="store_true", help="Run training across all session files")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print training logs per epoch step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 1. Load label.mat
    label_path = os.path.join(args.dataset_dir, "label.mat")
    if not os.path.exists(label_path):
        print(f"Error: label.mat not found at {label_path}")
        return
        
    labels_mat = sio.loadmat(label_path)
    # The label mapping: SEED labels in label.mat are [[1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]] (shape: 1, 15)
    # We map [-1, 0, 1] to [0, 1, 2] by adding 1.
    raw_labels = labels_mat['label'][0]
    mapped_labels = raw_labels + 1
    
    # 2. Get list of files to process
    if args.session:
        files_to_process = [args.session]
    elif args.run_all:
        # Get all subject .mat files in the directory
        files_to_process = sorted([f for f in os.listdir(args.dataset_dir) if f.endswith(".mat") and f != "label.mat"])
    else:
        # Default: Process only the first session files for all 15 subjects (as used in the paper's main table)
        # We find the chronologically first file for each subject (1 to 15)
        import re
        all_files = os.listdir(args.dataset_dir)
        pattern = re.compile(r"^(\d+)_(\d+)\.mat$")
        subj_sessions = {}
        for f in all_files:
            m = pattern.match(f)
            if m:
                subj = int(m.group(1))
                date = int(m.group(2))
                if subj not in subj_sessions:
                    subj_sessions[subj] = []
                subj_sessions[subj].append((date, f))
        
        files_to_process = []
        for subj in sorted(subj_sessions.keys()):
            sessions = sorted(subj_sessions[subj])
            files_to_process.append(sessions[0][1])
            
    print(f"Processing {len(files_to_process)} session(s): {files_to_process}")
    
    accuracies = []
    for idx, fname in enumerate(files_to_process):
        mat_path = os.path.join(args.dataset_dir, fname)
        print(f"\n==================================================")
        print(f"[{idx+1}/{len(files_to_process)}] Session File: {fname}")
        print(f"==================================================")
        
        best_acc = train_and_evaluate(mat_path, mapped_labels, args)
        accuracies.append(best_acc)
        print(f"-> Best Test Accuracy for {fname}: {best_acc*100:.2f}%")
        
    print("\n==================================================")
    print("FINAL SUMMARY STATISTICS")
    print("==================================================")
    for fname, acc in zip(files_to_process, accuracies):
        print(f"{fname:20s}: {acc*100:.2f}%")
    print("-" * 50)
    print(f"Number of Sessions: {len(accuracies)}")
    print(f"Average Accuracy  : {np.mean(accuracies)*100:.2f}%")
    print(f"Standard Deviation: {np.std(accuracies)*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
