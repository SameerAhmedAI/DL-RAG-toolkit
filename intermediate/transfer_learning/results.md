# Transfer Learning — Results

## Dataset
CIFAR-10, subsetted for CPU training time: 3,400 train images (2,800 train / 600 val after 85/15 split), 1,000 test images.
- Full CIFAR-10 (50k train / 10k test) available but not used — same CPU/timeline tradeoff as the CNN task
- Images resized 32x32 → 224x224 to match ResNet18's expected input, normalized using ImageNet mean/std (required since we're using ImageNet pretrained weights)

## Architecture
Pretrained **ResNet18** (ImageNet1K weights) as backbone:
- All layers frozen except `layer4` (final residual block) and the replaced classifier head
- Trainable parameters: **8,398,858 / 11,181,642 (75.1%)**
- Classifier head replaced: `Linear(512, 10)` for CIFAR-10's 10 classes

- Loss: CrossEntropyLoss
- Optimizer: Adam, lr=1e-4 (lower than from-scratch training, standard for fine-tuning)
- Epochs: 6, Batch size: 32
- Hardware: CPU

## Results
- Test Accuracy: **0.8410**
- Test Precision (weighted): **0.8425**
- Test Recall (weighted): **0.8410**
- Test F1 (weighted): **0.8409**

Per-class F1 ranged from 0.71 (cat, dog) to 0.94 (automobile).

See `results_artifacts/training_curves.png` and `results_artifacts/confusion_matrix.png`.

## Observations
- **Convergence was very fast**: train accuracy hit 93% by epoch 2 and 99%+ by epoch 3 — expected, since the pretrained ImageNet features already encode strong general visual representations (edges, textures, shapes), so the model only needs to learn the mapping to CIFAR-10's specific 10 classes rather than visual features from scratch.
- **Clear overfitting from epoch 3 onward**: train loss collapsed toward 0 (0.10 → 0.02) while val loss plateaued and slightly worsened (0.4767 → 0.4830). Val accuracy also plateaued (~84-86%) despite train accuracy sitting near-perfect. With 75.1% of parameters trainable and only 2,800 training images, the model has more than enough capacity to memorize the training set well before it stops improving on unseen data.
- **cat (71% F1) and dog (71% F1) were the weakest classes** — a well-known CIFAR-10 hard pair, as cats and dogs share more visual overlap (fur texture, four-legged pose, similar backgrounds) than classes like automobile/truck/ship which have more distinct shapes.
- Despite the overfitting, test accuracy (84.1%) is strong for only 2,800 training images — this is the core value proposition of transfer learning: the pretrained backbone made high accuracy achievable with a fraction of the data a from-scratch CNN would need.

## Challenges & Solutions
- **Overfitting given small dataset + high trainable parameter count (75.1%)**: identified via the train/val loss divergence starting epoch 3. Not corrected in this run — the result is documented honestly rather than re-tuned to hide it. A future iteration would freeze layer4 as well (leaving only the ~5K-parameter classifier head trainable) or add dropout before the final linear layer, trading some accuracy ceiling for better generalization.
- **DNS resolution failure on first attempt** downloading ResNet18 pretrained weights from `download.pytorch.org` — transient network issue (CIFAR-10 download from a different host succeeded in the same session). Resolved by retrying; no code change needed. Worth noting in the setup guide as a known flaky point for anyone reproducing this.