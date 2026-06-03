import os
import argparse
import re
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

def load_session_full_data(mat_path, label_arr, feature_type="LDS"):
    """
    Loads all 15 trials of a single session.
    Returns:
        trials_x: list of np.ndarray, each of shape (num_samples, 62, 10)
        trials_y: list of np.ndarray, each of shape (num_samples,)
    """
    data = sio.loadmat(mat_path)
    trials_x = []
    trials_y = []
    
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
        
        # Concatenate DE and PSD features along the frequency band dimension (axis=2)
        # Results in (num_samples, 62, 10)
        trial_feat = np.concatenate([de_feat, psd_feat], axis=-1)
        
        num_samples = trial_feat.shape[0]
        trial_label = label_arr[trial_idx - 1]
        trial_labels = np.full(num_samples, trial_label, dtype=np.int64)
        
        trials_x.append(trial_feat)
        trials_y.append(trial_labels)
        
    return trials_x, trials_y

def load_train_data_for_loso(train_session_paths, label_arr, feature_type="LDS"):
    """
    Loads and concatenates all 15 trials from multiple training sessions.
    """
    train_x = []
    train_y = []
    for path in train_session_paths:
        tx, ty = load_session_full_data(path, label_arr, feature_type)
        train_x.extend(tx)
        train_y.extend(ty)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    return train_x, train_y

def evaluate_with_voting(model, test_session_path, label_arr, train_mean, train_std, args):
    """
    Evaluates the model on the test session using a trial-level voting mechanism
    AND also computes the raw window-level accuracy.
    """
    model.eval()
    test_trials_x_np, test_trials_y_np = load_session_full_data(
        test_session_path, label_arr, feature_type=args.feature_type
    )
    
    correct_trials = 0
    total_trials = len(test_trials_x_np) # 15
    
    correct_windows = 0
    total_windows = 0
    
    with torch.no_grad():
        for t_idx in range(total_trials):
            x_trial = torch.tensor(test_trials_x_np[t_idx], dtype=torch.float32)
            y_trial = torch.tensor(test_trials_y_np[t_idx], dtype=torch.long)
            
            # Normalize with training statistics
            x_trial = (x_trial - train_mean) / train_std
            
            # Run model (on GPU)
            inputs = x_trial.cuda()
            outputs, _ = model(inputs)
            
            # Get predicted classes
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            # Window-level accuracy computation
            true_labels_np = y_trial.numpy()
            correct_windows += np.sum(preds == true_labels_np)
            total_windows += len(preds)
            
            # Majority voting for trial-level
            predicted_label = np.argmax(np.bincount(preds))
            true_label = true_labels_np[0]
            
            if predicted_label == true_label:
                correct_trials += 1
                
    voting_acc = correct_trials / total_trials
    window_acc = correct_windows / total_windows
    return voting_acc, window_acc

def train_and_evaluate_fold(train_session_paths, test_session_path, label_arr, fold_idx, args):
    # 1. Load Train Data
    train_x_np, train_y_np = load_train_data_for_loso(
        train_session_paths, label_arr, feature_type=args.feature_type
    )
    
    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.long)
    
    # 2. Compute Normalization Statistics on Train Data
    train_mean = train_x.mean(dim=0, keepdim=True)
    train_std = train_x.std(dim=0, keepdim=True) + 1e-6
    
    train_x = (train_x - train_mean) / train_std
    
    # 3. Create PyTorch DataLoader for Training
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Initialize Model
    model = DAGCN('seed').cuda()
    
    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
    best_voting_acc = 0.0
    best_window_acc = 0.0
    
    # 5. Training Loop
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
        
        # 6. Evaluation with Trial-Level Voting & Window Accuracy
        voting_acc, window_acc = evaluate_with_voting(
            model, test_session_path, label_arr, train_mean, train_std, args
        )
        
        if voting_acc > best_voting_acc:
            best_voting_acc = voting_acc
            
        if window_acc > best_window_acc:
            best_window_acc = window_acc
            
        if args.verbose and (epoch % 20 == 0 or epoch == args.epochs or epoch == 1):
            print(f"Fold {fold_idx} | Epoch {epoch:3d}/{args.epochs}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}%, Test Voting Acc = {voting_acc*100:.2f}%, Test Window Acc = {window_acc*100:.2f}%")
            
    return best_voting_acc, best_window_acc

def main():
    parser = argparse.ArgumentParser(description="DAGCN on SEED Cross-Session LOSO with Voting")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\Dev\BCI\EEG_Dataset\SEED\SEED\SEED_EEG\ExtractedFeatures_1s", help="Path to SEED ExtractedFeatures_1s directory")
    parser.add_argument("--subject", type=int, default=1, choices=range(1, 16), help="Subject ID (1 to 15) to run cross-session evaluation on")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay regularization")
    parser.add_argument("--feature_type", type=str, default="LDS", choices=["LDS", "movingAve", "raw"], help="Feature smoothing type")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type")
    parser.add_argument("--run_all", action="store_true", help="Run training across all subjects (1 to 15)")
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
    raw_labels = labels_mat['label'][0]
    mapped_labels = raw_labels + 1
    
    # 2. Get list of subjects to process
    subjects_to_process = list(range(1, 16)) if args.run_all else [args.subject]
    
    # 3. Group files by subject and sort chronologically
    all_files = os.listdir(args.dataset_dir)
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
            
    # Run Leave-One-Session-Out Cross-Validation
    subject_voting_accuracies = {}
    subject_window_accuracies = {}
    
    print(f"Running Cross-Session LOSO CV with Voting for Subject(s): {subjects_to_process}")
    
    for subj in subjects_to_process:
        if subj not in subj_sessions:
            print(f"Warning: No session files found for Subject {subj}")
            continue
            
        # Sort sessions chronologically (earliest to latest date)
        sessions = [os.path.join(args.dataset_dir, item[1]) for item in sorted(subj_sessions[subj])]
        if len(sessions) != 3:
            print(f"Warning: Subject {subj} does not have exactly 3 session files. Found {len(sessions)}. Skipping.")
            continue
            
        print(f"\n==================================================")
        print(f"Subject {subj} | Sessions: {[os.path.basename(s) for s in sessions]}")
        print(f"==================================================")
        
        fold_voting_accs = []
        fold_window_accs = []
        
        # 3-Fold Cross-Validation (leaving one session out)
        for fold in range(3):
            test_session = sessions[fold]
            train_sessions = [sessions[i] for i in range(3) if i != fold]
            
            print(f"\n--- Fold {fold} ---")
            print(f"Train on: {[os.path.basename(s) for s in train_sessions]}")
            print(f"Test on: {os.path.basename(test_session)} (with trial voting)")
            
            voting_acc, window_acc = train_and_evaluate_fold(
                train_sessions, test_session, mapped_labels, fold, args
            )
            fold_voting_accs.append(voting_acc)
            fold_window_accs.append(window_acc)
            print(f"-> Fold {fold} Best Test Voting Acc: {voting_acc*100:.2f}%, Best Test Window Acc: {window_acc*100:.2f}%")
            
        mean_voting = np.mean(fold_voting_accs)
        mean_window = np.mean(fold_window_accs)
        subject_voting_accuracies[subj] = mean_voting
        subject_window_accuracies[subj] = mean_window
        print(f"\n-> Subject {subj} Average Voting Accuracy: {mean_voting*100:.2f}%")
        print(f"-> Subject {subj} Average Window Accuracy: {mean_window*100:.2f}%")
        print(f"Fold Voting Accuracies: {[f'{acc*100:.2f}%' for acc in fold_voting_accs]}")
        print(f"Fold Window Accuracies: {[f'{acc*100:.2f}%' for acc in fold_window_accs]}")
        
    print("\n==================================================")
    print("FINAL LOSO CROSS-SESSION EVALUATION SUMMARY")
    print("==================================================")
    for subj in subject_voting_accuracies.keys():
        v_acc = subject_voting_accuracies[subj]
        w_acc = subject_window_accuracies[subj]
        print(f"Subject {subj:2d} | Mean Voting Acc: {v_acc*100:.2f}% | Mean Window Acc: {w_acc*100:.2f}%")
    print("-" * 60)
    print(f"Number of Subjects : {len(subject_voting_accuracies)}")
    print(f"Average Voting Acc : {np.mean(list(subject_voting_accuracies.values()))*100:.2f}%")
    print(f"Average Window Acc : {np.mean(list(subject_window_accuracies.values()))*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
