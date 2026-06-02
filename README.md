# DBGC-ATFFNet-AFTL

Code for the paper *A Dual-Branch Dynamic Graph Convolution Based Adaptive TransFormer Feature Fusion Network for EEG Emotion Recognition*.

## 實驗結果對比 (SEED Dataset)

以下是使用 DAGCN 模型在 SEED 資料集上，採用三種不同訓練與評估方法得出的準確率對比：

| 評估方法 | 訓練資料範疇 | 驗證集劃分方式 | 評估顆粒度 | 訓練 Epochs | 全體平均準確率 (Avg Acc) | 說明 / 特點 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **會話內試驗分割**<br>([train.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train.py)) | 單一 Session (S1) | 前 9 個 Trial 訓練，<br>後 6 個 Trial 測試 | Window 級 | 200 | **86.63%**<br>(Std: 8.38%) | 論文 Table 2 基線標準協議。 |
| **跨會話留一驗證**<br>([train_loso.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train_loso.py)) | 跨 Session (S1-S3) | 3-Fold 留一 Session 交叉驗證 | Window 級<br><hr>**Trial 級 (Voting)** | 10 | **67.75%**<br><hr>**70.96%** | 跨會話特徵偏移大，難度最高。<br>影片投票 (Voting) 可過濾 Window 雜訊。 |
| **會話內 5-Fold 隨機分割**<br>([train_5fold.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train_5fold.py)) | 單一 Session (S1) | 隨機打散 Window，<br>按 8:2 進行 5-Fold 交叉驗證 | Window 級 | 30 | **100.00%**<br>(Std: 0.00%) | 隨機 Window 切分會導致同 Trial 時間高度相關樣本洩漏 (Data Leakage)。 |
| **適配器微調遷移學習 (AFTL)**<br>([train_aftl.py](file:///c:/Dev/BCI/DBGC-ATFFNet-AFTL/train_aftl.py)) | 跨受試者 (Session 1) | 14個受試者預訓練模型，目標受試者按 5:5 分割微調 | **Window 級 (Shuffle)**<br><hr>**Trial 級 (No Leakage)** | Pretrain: 30<br>Finetune: 50 | **100.00%** (S1, bs=128)<br>**94.40%** (S1, bs=256)<br><hr>**61.28%** (S1, Best)<br>**26.22%** (S1, Final) | 鎖定 Model Backbone 只更新 1,456 個 Adapter 參數。<br>Window 級隨機分割存在嚴重的時間相關資料洩漏。<br>Trial 級無洩漏，但極易發生過度擬合 (Overfitting)。 |

## 快速使用指南

* **會話內試驗分割 (9-6 Split)**:
  ```bash
  python train.py --run_all --epochs 200
  ```
* **跨會話留一驗證 (LOSO with Voting & Window)**:
  ```bash
  python train_loso.py --run_all --epochs 10
  ```
* **會話內 5-Fold 隨機分割**:
  ```bash
  python train_5fold.py --run_all --epochs 30
  ```
* **跨受試者 AFTL 遷移學習 (以 Subject 1 為目標受試者，Window 級分割)**:
  ```bash
  python train_aftl.py --target_subject 1 --pretrain_epochs 30 --finetune_epochs 50 --split_type window
  ```
* **跨受試者 AFTL 遷移學習 (以 Subject 1 為目標受試者，Trial 級分割)**:
  ```bash
  python train_aftl.py --target_subject 1 --pretrain_epochs 30 --finetune_epochs 50 --split_type trial
  ```
* **跨受試者 AFTL 遷移學習 (跑完所有 15 個目標受試者)**:
  ```bash
  python train_aftl.py --run_all --pretrain_epochs 30 --finetune_epochs 50 --split_type window
  ```


