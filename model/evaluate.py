import torch
import torch.nn as nn
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
from pathlib import Path
from helpers import collate_fn_test, idx2char, blank, chars, spec_transform
from model import QuartzNetBxR
import os
from dotenv import load_dotenv

load_dotenv()
root = str(os.getenv("ROOT"))

# ---------------------------------------------------------------------------
# Greedy CTC decoder
# ---------------------------------------------------------------------------

def _ctc_greedy_decode(token_ids):
    decoded = []
    prev = None
    for token in token_ids:
        if token != blank and token != prev:
            decoded.append(idx2char[token])
        prev = token
    return "".join(decoded)


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------

def _word_edit_distance(ref_words, hyp_words):
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1]


def _compute_wer(refs, hyps):
    total_errors = 0
    total_words  = 0
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_errors += _word_edit_distance(ref_words, hyp_words)
        total_words  += len(ref_words)
    return (total_errors / max(1, total_words)) * 100


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = QuartzNetBxR(
        n_mels=config.get("n_mels", 64),
        n_classes=config.get("n_classes", 29),
        B=config.get("B", 5),
        R=config.get("R", 5),
    ).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Build LM decoder
# ---------------------------------------------------------------------------

def _build_lm_decoder(arpa_path, alpha=0.5, beta=1.5):
    from pyctcdecode import build_ctcdecoder
    vocab = list(chars)
    vocab.append("")
    return build_ctcdecoder(
        vocab,
        str(arpa_path),
        unigrams=None,
        alpha=alpha,
        beta=beta,
    )


# ---------------------------------------------------------------------------
# Evaluate one dataset split
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, device, lm_decoder=None, beam_width=100, split_name=""):
    greedy_refs, greedy_hyps = [], []
    lm_refs,     lm_hyps     = [], []

    total_batches = len(loader)
    print(f"\n  Evaluating {split_name} ({total_batches} batches)...")

    with torch.no_grad():
        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(loader):
            inputs         = inputs.to(device)
            targets_cpu    = targets.cpu()
            target_lengths_cpu = target_lengths.cpu()

            logits = model(inputs)  # (batch, n_classes, time)

            # --- greedy ---
            pred_ids = logits.argmax(dim=1).cpu()
            for i in range(pred_ids.size(0)):
                hyp = _ctc_greedy_decode(pred_ids[i].tolist())
                tlen = int(target_lengths_cpu[i].item())
                ref = "".join(idx2char[t] for t in targets_cpu[i, :tlen].tolist())
                greedy_hyps.append(hyp)
                greedy_refs.append(ref)

            # --- LM beam search ---
            if lm_decoder is not None:
                log_probs_batch = logits.cpu().log_softmax(dim=1)
                for i in range(log_probs_batch.size(0)):
                    lp = log_probs_batch[i].T.numpy()  # (time, n_classes)
                    hyp = lm_decoder.decode(lp, beam_width=beam_width)
                    tlen = int(target_lengths_cpu[i].item())
                    ref = "".join(idx2char[t] for t in targets_cpu[i, :tlen].tolist())
                    lm_hyps.append(hyp)
                    lm_refs.append(ref)

            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                print(f"    {batch_idx+1}/{total_batches} batches done...")

    greedy_wer = _compute_wer(greedy_refs, greedy_hyps)
    lm_wer     = _compute_wer(lm_refs, lm_hyps) if lm_decoder is not None else None
    return greedy_wer, lm_wer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path("/home/xz/GOATS422/Notarius/outputs/checkpoints/best.pt")
    arpa_path       = Path("/home/xz/GOATS422/Notarius/model/lm/3-gram.pruned.3e-7.arpa")
    arpa_path_2     = Path("/home/xz/GOATS422/Notarius/model/lm/3-gram.pruned.1e-7.arpa")

    print(f"Loading model from {checkpoint_path}...")
    model = _load_model(checkpoint_path, device)

    print("Loading LM decoders...")
    lm_decoder_3e7 = _build_lm_decoder(arpa_path,   alpha=0.5, beta=1.5)
    lm_decoder_1e7 = _build_lm_decoder(arpa_path_2, alpha=0.5, beta=1.5)

    # Datasets
    test_clean = LIBRISPEECH(root=root, url="test-clean", download=False)
    

    loader_clean = DataLoader(test_clean, batch_size=32, shuffle=False,
                              collate_fn=collate_fn_test, num_workers=4)
    

    # Evaluate
    clean_greedy, clean_lm_3e7 = evaluate_split(model, loader_clean, device, lm_decoder_3e7, split_name="test-clean (3e-7 LM)")
    clean_greedy, clean_lm_1e7 = evaluate_split(model, loader_clean, device, lm_decoder_1e7, split_name="test-clean (1e-7 LM)")
    other_greedy, other_lm_3e7 = evaluate_split(model, loader_other, device, lm_decoder_3e7, split_name="test-other (3e-7 LM)")
    other_greedy, other_lm_1e7 = evaluate_split(model, loader_other, device, lm_decoder_1e7, split_name="test-other (1e-7 LM)")

    # Results table
    print("\n")
    print("=" * 75)
    print(f"  Table: LibriSpeech results, WER (%)")
    print("=" * 75)
    print(f"  {'Model':<20} {'Augment':<28} {'LM':<12} {'Test clean':>10} {'Test other':>10}")
    print("-" * 75)
    aug = "SpecAugment+speed perturb"
    print(f"  {'QuartzNet 5x5':<20} {aug:<28} {'None':<12} {clean_greedy:>10.2f} {other_greedy:>10.2f}")
    print(f"  {'QuartzNet 5x5':<20} {aug:<28} {'3-gram(3e-7)':<12} {clean_lm_3e7:>10.2f} {other_lm_3e7:>10.2f}")
    print(f"  {'QuartzNet 5x5':<20} {aug:<28} {'3-gram(1e-7)':<12} {clean_lm_1e7:>10.2f} {other_lm_1e7:>10.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()