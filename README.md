# DBGC-ATFFNet-AFTL

PyTorch implementation for the paper *A Dual-Branch Dynamic Graph Convolution Based Adaptive TransFormer Feature Fusion Network for EEG Emotion Recognition*.

## Experimental Results (SEED Dataset)

Below is the comparison of classification performance using the `DAGCN` model on the SEED dataset across four training/evaluation protocols:

| Evaluation Protocol | Data Scope | Validation Split | Granularity | Training Epochs | Avg Accuracy (Subject 1 / All) | Key Characteristics & Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Within-Session Subject-Dependent**<br>([train.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train.py)) | Single Session (S1) | Trials 1-9 for training,<br>trials 10-15 for testing | Window-level | 200 | **86.63%** (Avg)<br>(Std: 8.38%) | The baseline protocol reported in Table 2 of the paper. |
| **Cross-Session LOSO with Voting**<br>([train_loso.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train_loso.py)) | Cross-Session (S1-S3) | 3-Fold Leave-One-Session-Out | Window-level /<br>Trial-level (Voting) | 10 | **67.75%** (Window)<br>**70.96%** (Voting) | Hardest setting due to inter-session feature shifts. Majority voting effectively filters window-level noise. |
| **Within-Session 5-Fold CV**<br>([train_5fold.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train_5fold.py)) | Single Session (S1) | Randomly shuffle and split all windows 8:2 | Window-level | 30 | **100.00%** (Avg)<br>(Std: 0.00%) | Random window-level splitting on a single session suffers from massive temporal correlation data leakage. |
| **Adapter-Finetuned Transfer (AFTL)**<br>([train_aftl.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train_aftl.py)) | Cross-Subject (S1) | Pretrain on 14 source subjects;<br>Fine-tune on 50% target data | **Window-level (Shuffle)**<br><hr>**Trial-level (No Leakage)** | Pretrain: 30<br>Finetune: 50 | **100.00%** (S1, bs=128)<br>**94.40%** (S1, bs=256)<br><hr>**61.28%** (S1, Best)<br>**26.22%** (S1, Final) | Freezes the backbone and only updates the 1,456 Adapter parameters. Window-split suffers from data leakage (matches the paper). Trial-split is strictly leakage-free but prone to overfitting. |

---

## Quick Start Guide

### 1. Requirements & Dependencies
* Python 3.9+
* PyTorch (with CUDA support)
* SciPy, NumPy, Matplotlib

### 2. Running the Code

* **Within-Session Subject-Dependent (9-6 Split)**:
  ```bash
  python train.py --run_all --epochs 200
  ```
* **Cross-Session Leave-One-Session-Out (LOSO) with Voting**:
  ```bash
  python train_loso.py --run_all --epochs 10
  ```
* **Within-Session 5-Fold Cross-Validation**:
  ```bash
  python train_5fold.py --run_all --epochs 30
  ```
* **Adapter-Finetuned Transfer Learning (AFTL)**:
  * Run target subject 1 transfer learning (Window-level Split, with leakage):
    ```bash
    python train_aftl.py --target_subject 1 --pretrain_epochs 30 --finetune_epochs 50 --split_type window
    ```
  * Run target subject 1 transfer learning (Trial-level Split, leakage-free):
    ```bash
    python train_aftl.py --target_subject 1 --pretrain_epochs 30 --finetune_epochs 50 --split_type trial
    ```
  * Run cross-subject AFTL transfer evaluation across all 15 target subjects:
    ```bash
    python train_aftl.py --run_all --pretrain_epochs 30 --finetune_epochs 50 --split_type window
    ```
