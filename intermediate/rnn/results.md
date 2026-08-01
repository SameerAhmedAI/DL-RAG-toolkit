# RNN — Results

## Dataset
20 Newsgroups, 4-category subset: `comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.guns`.
- Headers/footers/quotes stripped to prevent trivial leakage (e.g. mailing list signatures leaking the category)
- Texts shorter than 20 characters after stripping filtered out
- Split: 70% train / 15% val / 15% test, stratified — 2,608 train / 559 val / 559 test
- Custom vocabulary (10,002 tokens, min frequency 2), sequences truncated/padded to 100 tokens

## Architecture
`Embedding(10002, 100) → RNN(hidden=128, tanh) → Dropout(0.3) → Linear(128, 4)`

- Loss: CrossEntropyLoss
- Optimizer: Adam, lr=1e-3
- Gradient clipping (max_norm=5.0) — standard for RNN training stability
- Epochs: 10, Batch size: 32
- Hardware: CPU

## Results
- Test Accuracy: **0.2719** (near the 25% random-guess baseline for 4 classes)
- Test Precision (weighted): **0.2628**
- Test Recall (weighted): **0.2719**
- Test F1 (weighted): **0.2589**

Per-class F1 ranged 0.12–0.34 — no class was learned well.

See `results_artifacts/training_curves.png` and `results_artifacts/confusion_matrix.png`.

## Observations
- **Clear overfitting from epoch 2 onward**: train accuracy climbed steadily to 59.5%, while val accuracy stayed flat near the random-guess baseline (26–29%) and val loss *increased every epoch* after epoch 1 (1.38 → 2.05). The model memorized training examples without learning generalizable patterns.
- This is the textbook vanishing-gradient failure mode of vanilla RNNs on longer sequences (100 tokens here): gradients from later timesteps struggle to propagate back to update earlier-layer weights meaningfully, so the network can't learn long-range dependencies in the text.
- `rec.sport.baseball` had the worst recall (0.09) — the model essentially failed to identify this class at test time despite balanced training data, consistent with the model learning noise rather than signal.
- This result is intentionally paired with the LSTM model on identical data/hyperparameters (see `intermediate/lstm/results.md`) — the comparison is the point, not a tuning failure to fix. LSTM on the same setup nearly doubled test accuracy (54.9% vs 27.2%), directly demonstrating why gated architectures replaced vanilla RNNs for sequence tasks.

## Challenges & Solutions
- **Overfitting / poor generalization**: identified as an architectural limitation (vanishing gradients over 100-token sequences), not a bug or hyperparameter issue — confirmed by the LSTM comparison on identical data producing a stable, generalizing model. No further RNN tuning was pursued, since the comparison itself is the deliverable — demonstrating *why* LSTM exists is more valuable here than forcing the RNN to a higher (and less honest) number.