# ANN — Results

## Dataset
Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`.
- 569 samples, 30 numeric features, 2 classes (malignant / benign)
- Split: 70% train / 15% val / 15% test, stratified
- Features standardized (zero mean, unit variance) using training-set statistics only

## Architecture
Feedforward ANN: `Input(30) → Linear(64) → BatchNorm → ReLU → Dropout(0.3) → Linear(32) → BatchNorm → ReLU → Dropout(0.3) → Linear(2)`

- Loss: CrossEntropyLoss
- Optimizer: Adam, lr=1e-3
- Epochs: 30
- Batch size: 32
- Hardware: CPU (no GPU available)

## Results
- Test Accuracy: **0.9767**
- Test Precision (weighted): **0.9776**
- Test Recall (weighted): **0.9767**
- Test F1 (weighted): **0.9766**

Per-class:
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Malignant | 1.00 | 0.94 | 0.97 | 32 |
| Benign | 0.96 | 1.00 | 0.98 | 54 |

See `results_artifacts/training_curves.png` and `results_artifacts/confusion_matrix.png`.

## Observations
- Train and validation loss decreased together throughout training with no divergence — no overfitting, despite only 398 training samples. Dropout (0.3) and BatchNorm likely contributed to this given the small dataset size.
- Validation accuracy plateaued around epoch 15 (~97.6%) while train accuracy kept climbing slightly — a mild but expected gap, not concerning at this scale.
- The model is more conservative on malignant cases (100% precision, 94% recall) than benign (96% precision, 100% recall) — i.e., it occasionally misses a malignant case rather than over-flagging benign ones as malignant. In a real diagnostic context this would be the wrong direction to optimize for (false negatives on malignant are more costly), worth noting even though this is a demo, not a deployed system.

## Challenges & Solutions
- None encountered — dataset is small, clean, and well-separated (a standard sklearn benchmark), so no data quality or convergence issues arose. Main design decision was a stratified split to preserve class balance given the dataset's inherent imbalance (212 malignant vs 357 benign).