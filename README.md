# IBNet: Inverted Bottleneck Network for Lightweight ASR

IBNet is an end-to-end convolutional neural network for automatic speech recognition (ASR) that integrates inverted bottleneck modules into the time-channel separable convolution framework of QuartzNet. It is trained with CTC loss and evaluated on LibriSpeech.

## Architecture

IBNet keeps the same macro-structure as QuartzNet (C1 → B1–B5 → C2 → C3 → C4) but replaces every convolution module with an inverted bottleneck variant and adds a dual residual connection scheme.

### IBConv Module

Each `IBConv` expands channels before the depthwise conv and compresses back after, following the MobileNetV2 inverted bottleneck pattern:

```
Pointwise expand (C → C×t) → BN → ReLU
Depthwise Conv               → BN → ReLU
Pointwise compress (C×t → C) → BN
```

The final compress step has **no ReLU** (linear bottleneck) — in the narrow output space, ReLU would discard roughly half the activations and irreversibly lose information.

When input and output channels match and stride=1, each `IBConv` adds a per-module skip connection. Combined with the existing block-level residual, this forms a **dual residual scheme** that provides two complementary gradient pathways through the network.

### IBBlock

Each `IBBlock` stacks R `IBConv` modules. The first module handles any channel change; the remaining R−1 operate at fixed width and each benefit from the per-module residual. A block-level residual (1×1 conv + BN) connects the block input directly to the block output.

### Overall Structure

| Layer | Description |
|-------|-------------|
| C1 | Conv1d k=33, stride=2, BN, ReLU |
| B1 | IBBlock C→C, k=33 |
| B2 | IBBlock C→C, k=39 |
| B3 | IBBlock C→2C, k=51 |
| B4 | IBBlock 2C→2C, k=63 |
| B5 | IBBlock 2C→2C, k=75 |
| C2 | IBConv 2C→2C, k=87 |
| C3 | Conv1d 2C→4C, k=1, BN, ReLU |
| C4 | Conv1d 4C→classes, k=1, dilation=2 |

Blocks B1–B2 maintain C channels; B3–B5 double to 2C. This follows the established principle that early layers extract low-level acoustic features (adequate with fewer channels) while deeper layers capture higher-level linguistic abstractions (requiring a richer feature space).

### Configurable Channel Width

A single `C` parameter controls model size:

| C | Params | Comparable to |
|---|--------|---------------|
| 172 | ~6.2M | QuartzNet 5×5 (6.7M) |
| 192 | ~8.2M | between QuartzNet 5×5 and 10×5 |
| 256 | ~14.1M | QuartzNet 10×5–15×5 |

## Results

LibriSpeech WER (%) — all models trained on `train-clean-100`:

| Model | Config | Params (M) | Dev clean | Dev other | Test clean | Test other |
|-------|--------|------------|-----------|-----------|------------|------------|
| QuartzNet 5x5 | Greedy | 6.7 | 50.63 | 67.73 | 50.43 | 68.18 |
| QuartzNet 5x5 | SpecCutout | 6.7 | 43.62 | 61.89 | 43.58 | 62.30 |
| QuartzNet 5x5 | Speed Perturb | 6.7 | 33.53 | 52.44 | 33.48 | 53.36 |
| QuartzNet 5x5 | SpecCutout + LM | 6.7 | 28.72 | 46.88 | 28.40 | 48.05 |
| QuartzNet 10x5 | Greedy | 12.8 | 35.43 | 53.99 | 35.06 | 55.01 |
| QuartzNet 10x5 | SpecCutout | 12.8 | 38.76 | 57.10 | 38.96 | 57.77 |
| QuartzNet 10x5 | Speed Perturb | 12.8 | 31.89 | 50.20 | 31.64 | 51.06 |
| QuartzNet 10x5 | SpecCutout + LM | 12.8 | 25.00 | 42.70 | 25.20 | 43.70 |
| IBNet (C=192) | Greedy | 8.2 | 29.07 | 48.02 | 29.15 | 49.30 |
| IBNet (C=192) | SpecCutout | 8.2 | 24.81 | 43.44 | 24.77 | 44.11 |
| IBNet (C=192) | Speed Perturb | 8.2 | 21.83 | 40.28 | 21.96 | 40.95 |
| IBNet (C=192) | SpecCutout + LM | 8.2 | 17.02 | 33.38 | 17.02 | 33.90 |

## Training

```bash
# IBNet
torchrun --standalone --nproc_per_node=2 -m model.training.train_ibnet \
  --epochs 50 --R 3 --C 192 --expand 2 \
  --lr 0.005 --warmup 2 --batch-size 180 \
  --output-dir outputs/notarius

# QuartzNet
torchrun --standalone --nproc_per_node=2 -m model.training.train_qn \
  --epochs 50 --R 5 \
  --lr 0.005 --warmup 2 --batch-size 180 \
  --output-dir outputs/quartznet
```

Checkpoints are saved to `outputs/<run-id>/`:

- `last.pt` — most recent epoch
- `best.pt` — best validation WER
- `epoch_XXX.pt` — periodic snapshot (every 10 epochs)
- `final_model.pt` — final weights

Resume a stopped run:

```bash
torchrun --standalone --nproc_per_node=2 -m model.training.train_ibnet \
  --epochs 50 --output-dir outputs/notarius --resume last.pt
```

## Transcription

```bash
# Greedy decoding
python -m model.scripts.transcribe_lm --audio audio.wav --checkpoint outputs/notarius/<run-id>/best.pt

# With language model
python -m model.scripts.transcribe_lm --audio audio.wav \
  --checkpoint outputs/notarius/<run-id>/best.pt \
  --arpa model/lm/3-gram.pruned.3e-7.arpa
```
