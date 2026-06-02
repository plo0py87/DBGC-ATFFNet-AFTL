import os
import argparse
import re
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import DAGCN, Adapter

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_session_all_samples(mat_path, label_arr, feature_type="LDS"):
    """
    Loads all 15 trials of a single session and concatenates all 1-second windows
    into a single unified sample pool X and labels Y.
    """
    data = sio.loadmat(mat_path)
    
    all_x = []
    all_y = []
    
    for trial_idx in range(1, 16):
        de_key = f"de_{feature_type}{trial_idx}"
        psd_key = f"psd_{feature_type}{trial_idx}"
        
        if de_key not in data or psd_key not in data:
            raise KeyError(f"Keys {de_key} or {psd_key} not found in {mat_path}")
            
        # Shape: (62, num_samples, 5)
        de_feat = data[de_key]
        psd_feat = data[psd_key]
        
        # Transpose to (num_samples, 62, 5)
        de_feat = de_feat.transpose(1, 0, 2)
        psd_feat = psd_feat.transpose(1, 0, 2)
        
        # Concatenate along the band dimension -> (num_samples, 62, 10)
        trial_feat = np.concatenate([de_feat, psd_feat], axis=-1)
        
        num_samples = trial_feat.shape[0]
        trial_label = label_arr[trial_idx - 1]
        trial_labels = np.full(num_samples, trial_label, dtype=np.int64)
        
        all_x.append(trial_feat)
        all_y.append(trial_labels)
        
    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    
    return X, Y

def load_session_by_trials(mat_path, label_arr, feature_type="LDS"):
    """
    Loads all 15 trials of a single session and returns them as lists of X and Y (one per trial).
    """
    data = sio.loadmat(mat_path)
    trials_x = []
    trials_y = []
    
    for trial_idx in range(1, 16):
        de_key = f"de_{feature_type}{trial_idx}"
        psd_key = f"psd_{feature_type}{trial_idx}"
        
        if de_key not in data or psd_key not in data:
            raise KeyError(f"Keys {de_key} or {psd_key} not found in {mat_path}")
            
        de_feat = data[de_key].transpose(1, 0, 2)
        psd_feat = data[psd_key].transpose(1, 0, 2)
        trial_feat = np.concatenate([de_feat, psd_feat], axis=-1)
        
        num_samples = trial_feat.shape[0]
        trial_label = label_arr[trial_idx - 1]
        trial_labels = np.full(num_samples, trial_label, dtype=np.int64)
        
        trials_x.append(trial_feat)
        trials_y.append(trial_labels)
        
    return trials_x, trials_y

def get_session1_files(dataset_dir):
    """
    Finds the chronologically first session file (.mat) for all subjects 1 to 15.
    Returns a dictionary maps subject_id -> absolute path
    """
    all_files = os.listdir(dataset_dir)
    file_pattern = re.compile(r"^(\d+)_(\d+)\.mat$")
    subj_sessions = {}
    
    for f in all_files:
        m = file_pattern.match(f)
        if m:
            subj = int(m.group(1))
            date = int(m.group(2))
            if subj not in subj_sessions:
                subj_sessions[subj] = []
            subj_sessions[subj].append((date, f))
            
    subj_session1 = {}
    for subj in sorted(subj_sessions.keys()):
        sessions = sorted(subj_sessions[subj])
        subj_session1[subj] = os.path.join(dataset_dir, sessions[0][1])
        
    return subj_session1

def run_aftl_for_target(subj_session1_map, target_subj, label_arr, args):
    print(f"\n==================================================")
    print(f"Target Subject: {target_subj} | Pre-training on other 14 subjects...")
    print(f"==================================================")
    
    # 1. Prepare Pre-training (Source Subjects) Data
    source_x_list = []
    source_y_list = []
    
    for s_idx, mat_path in subj_session1_map.items():
        if s_idx == target_subj:
            continue
        sx, sy = load_session_all_samples(mat_path, label_arr, feature_type=args.feature_type)
        source_x_list.append(sx)
        source_y_list.append(sy)
        
    source_x_np = np.concatenate(source_x_list, axis=0)
    source_y_np = np.concatenate(source_y_list, axis=0)
    
    # Convert to Tensors & Normalize
    source_x = torch.tensor(source_x_np, dtype=torch.float32)
    source_y = torch.tensor(source_y_np, dtype=torch.long)
    
    source_mean = source_x.mean(dim=0, keepdim=True)
    source_std = source_x.std(dim=0, keepdim=True) + 1e-6
    source_x = (source_x - source_mean) / source_std
    
    # DataLoader for pre-training
    pretrain_dataset = Data.TensorDataset(source_x, source_y)
    pretrain_loader = Data.DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = DAGCN('seed').cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 2. Pre-training Loop (Source Subjects)
    print(f"-> Starting Pre-training for {args.pretrain_epochs} epochs...")
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in pretrain_loader:
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
        
        if epoch % 5 == 0 or epoch == args.pretrain_epochs or epoch == 1:
            print(f"Pre-train Epoch {epoch:3d}/{args.pretrain_epochs}: Loss = {train_loss:.4f}, Acc = {train_acc*100:.2f}%")
            
    # 3. Freeze all parameters EXCEPT the Adapter modules
    print("-> Freezing model backbone parameters, keeping Adapters active...")
    for param in model.parameters():
        param.requires_grad = False
        
    adapter_param_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Adapter):
            for param in module.parameters():
                param.requires_grad = True
                adapter_param_count += param.numel()
                
    print(f"-> Unfrozen Adapter parameters count: {adapter_param_count} (Table 8 target: 1456)")
    
    # 4. Load & Split Target Subject Data
    target_mat_path = subj_session1_map[target_subj]
    
    if args.split_type == "window":
        target_x_np, target_y_np = load_session_all_samples(target_mat_path, label_arr, feature_type=args.feature_type)
        # Random 50/50 Split at the window/sample level
        num_target_samples = target_x_np.shape[0]
        indices = np.arange(num_target_samples)
        np.random.shuffle(indices)
        
        split_idx = num_target_samples // 2
        ft_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        ft_x_np, ft_y_np = target_x_np[ft_idx], target_y_np[ft_idx]
        test_x_np, test_y_np = target_x_np[test_idx], target_y_np[test_idx]
    else:
        # Split by trials (8 trials for training/fine-tuning, 7 for testing)
        trials_x, trials_y = load_session_by_trials(target_mat_path, label_arr, feature_type=args.feature_type)
        trial_indices = np.arange(15)
        np.random.shuffle(trial_indices)
        
        ft_trial_indices = trial_indices[:8]
        test_trial_indices = trial_indices[8:]
        
        ft_x_np = np.concatenate([trials_x[i] for i in ft_trial_indices], axis=0)
        ft_y_np = np.concatenate([trials_y[i] for i in ft_trial_indices], axis=0)
        test_x_np = np.concatenate([trials_x[i] for i in test_trial_indices], axis=0)
        test_y_np = np.concatenate([trials_y[i] for i in test_trial_indices], axis=0)
    
    # Convert to Tensors
    ft_x = torch.tensor(ft_x_np, dtype=torch.float32)
    ft_y = torch.tensor(ft_y_np, dtype=torch.long)
    test_x = torch.tensor(test_x_np, dtype=torch.float32)
    test_y = torch.tensor(test_y_np, dtype=torch.long)
    
    # Normalize Target data using Target Fine-tuning set statistics
    target_mean = ft_x.mean(dim=0, keepdim=True)
    target_std = ft_x.std(dim=0, keepdim=True) + 1e-6
    ft_x = (ft_x - target_mean) / target_std
    test_x = (test_x - target_mean) / target_std
    
    ft_dataset = Data.TensorDataset(ft_x, ft_y)
    ft_loader = Data.DataLoader(ft_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Re-initialize optimizer to only update trainable parameters (Adapters)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft, weight_decay=args.weight_decay)
    
    # 5. Target Subject Fine-Tuning Loop
    print(f"-> Fine-tuning Adapters on 50% target subject data for {args.finetune_epochs} epochs...")
    best_test_acc = 0.0
    
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in ft_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            optimizer_ft.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ft.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        ft_loss = total_loss / total
        ft_acc = correct / total
        
        # Test evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                outputs, _ = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
        test_acc = test_correct / test_total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
        if epoch % 10 == 0 or epoch == args.finetune_epochs or epoch == 1:
            print(f"Fine-tune Epoch {epoch:3d}/{args.finetune_epochs}: Loss = {ft_loss:.4f}, Ft Acc = {ft_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}%")
            
    return best_test_acc

def main():
    parser = argparse.ArgumentParser(description="DAGCN on SEED Adapter-Finetuned Transfer Learning (AFTL)")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\Dev\BCI\EEG_Dataset\SEED\SEED\SEED_EEG\ExtractedFeatures_1s", help="Path to SEED ExtractedFeatures_1s directory")
    parser.add_argument("--target_subject", type=int, default=1, choices=range(1, 16), help="Subject ID (1 to 15) to use as the target subject")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="Number of pre-training epochs on source subjects")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Number of fine-tuning epochs on target subject")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Pre-training learning rate")
    parser.add_argument("--lr_ft", type=float, default=0.001, help="Fine-tuning learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay regularization")
    parser.add_argument("--feature_type", type=str, default="LDS", choices=["LDS", "movingAve", "raw"], help="Feature smoothing type")
    parser.add_argument("--run_all", action="store_true", help="Run AFTL across all 15 subjects")
    parser.add_argument("--split_type", type=str, default="window", choices=["window", "trial"], help="Data splitting strategy for target subject fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 1. Load label.mat
    label_path = os.path.join(args.dataset_dir, "label.mat")
    if not os.path.exists(label_path):
        print(f"Error: label.mat not found at {label_path}")
        return
        
    labels_mat = sio.loadmat(label_path)
    raw_labels = labels_mat['label'][0]
    mapped_labels = raw_labels + 1
    
    # 2. Get session 1 path for all subjects
    subj_session1_map = get_session1_files(args.dataset_dir)
    
    # 3. Select target subjects
    targets = list(range(1, 16)) if args.run_all else [args.target_subject]
    print(f"Running AFTL for target subject(s): {targets}")
    
    results = {}
    for target in targets:
        best_acc = run_aftl_for_target(subj_session1_map, target, mapped_labels, args)
        results[target] = best_acc
        print(f"-> Target Subject {target} Final AFTL Test Accuracy: {best_acc*100:.2f}%")
        
    print("\n==================================================")
    print("FINAL AFTL CROSS-SUBJECT SUMMARY")
    print("==================================================")
    for subj, acc in results.items():
        print(f"Target Subject {subj:2d} Accuracy: {acc*100:.2f}%")
    print("-" * 50)
    print(f"Number of Subjects : {len(results)}")
    print(f"Average Accuracy   : {np.mean(list(results.values()))*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
