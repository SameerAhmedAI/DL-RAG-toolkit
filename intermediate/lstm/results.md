# LSTM — Results

## Dataset
Identical to RNN — 20 Newsgroups, 4-category subset (`comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.guns`), same split, same vocabulary construction, same 100-token sequence length. Kept identical deliberately to isolate architecture as the only variable in the RNN vs. LSTM comparison.

## Architecture
`Embedding(10002, 100) → LSTM(hidden=128) → Dropout(0.3) → Linear(128, 4)`

- Loss: CrossEntropyLoss
- Optimizer: Adam, lr=1e-3 (identical to RNN)
- Gradient clipping (max_norm=5.0)
- Epochs: 10, Batch size: 32 (identical to RNN)
- Hardware: CPU

## Results
- Test Accuracy: **0.5492**
- Test Precision (weighted): **0.5808**
- Test Recall (weighted): **0.5492**
- Test F1 (weighted): **0.5498**

Per-class:
| Class | Precision | Recall | F1 |
|---|---|---|---|
| comp.graphics | 0.75 | 0.57 | 0.65 |
| rec.sport.baseball | 0.66 | 0.70 | 0.68 |
| sci.space | 0.52 | 0.33 | 0.40 |
| talk.politics.guns | 0.38 | 0.61 | 0.47 |

See `results_artifacts/training_curves.png` and `results_artifacts/confusion_matrix.png`.

## Observations
- **Roughly 2x the RNN's test accuracy on identical data and hyperparameters** (54.9% vs 27.2%) — the clearest possible demonstration that LSTM's gating mechanism (input/forget/output gates) mitigates the vanishing gradient problem that crippled the vanilla RNN on the same 100-token sequences.
- Val accuracy still fluctuated (peaked at 49.2% epoch 7, dropped to 38.3% epoch 9) and val loss didn't monotonically decrease — the model is still overfitting to some degree on only 2,608 training samples, just far less catastrophically than the RNN. More training data or regularization (higher dropout, weight decay) would likely narrow this further.
- `sci.space` had the weakest recall (0.33) despite reasonable precision (0.52) — the model under-predicts this class, likely confusing it with `comp.graphics` (both technical/scientific vocabulary might overlap more than sports or politics terminology does).
- `rec.sport.baseball` was the strongest class (0.68 F1) — sports vocabulary is likely the most lexically distinct from the other three categories.

## Challenges & Solutions
- **Residual overfitting despite the architectural improvement**: LSTM still shows val loss/accuracy instability late in training. Not corrected here since the primary goal — demonstrating the RNN-to-LSTM improvement — was achieved with the identical, controlled setup. Documented as a natural next step (more data, dropout tuning, or early stopping) rather than solved, since "solving" it would mean changing hyperparameters and breaking the direct RNN comparison.