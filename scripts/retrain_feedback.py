"""retrain_feedback.py — bake feedback.jsonl into a model checkpoint.

Converts approved/taught conversation pairs from feedback.jsonl into
a text corpus and fine-tunes a checkpoint on it.

Usage:
  python scripts/retrain_feedback.py --ckpt models/aetheria_soul.pt
  python scripts/retrain_feedback.py --ckpt models/aetheria_soul.pt --epochs 5
  python scripts/retrain_feedback.py --ckpt models/aetheria_soul.pt --out models/aetheria_soul.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prototype_model import (  # noqa: E402
    TinyTransformerLM, TextDataset, collate_fn, _save_checkpoint
)


def _load_feedback(feedback_path: Path) -> list[dict]:
    pairs = []
    if not feedback_path.exists():
        return pairs
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("human") and obj.get("aetheria"):
                pairs.append(obj)
        except Exception:
            pass
    return pairs


def _pairs_to_text(pairs: list[dict]) -> str:
    """Convert pairs to a training corpus."""
    blocks = []
    for p in pairs:
        blocks.append(f"Human: {p['human']}\nAetheria: {p['aetheria']}")
    return "\n\n".join(blocks)


def _load_ckpt(path: Path):
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    return ckpt


def finetune(ckpt_path: Path, feedback_path: Path, out_path: Path,
             spm_path: Path, epochs: int = 5, lr: float = 5e-5,
             batch_size: int = 4, seq_len: int = 64, device_str: str = None):

    # ── Load feedback ─────────────────────────────────────────────────────────
    pairs = _load_feedback(feedback_path)
    if not pairs:
        print("No feedback pairs found in", feedback_path)
        print("Have a conversation first and approve/teach replies.")
        return

    # weight taught > approved > forced for display
    by_source = {}
    for p in pairs:
        by_source.setdefault(p.get("source", "?"), []).append(p)
    print(f"Feedback loaded: {len(pairs)} pairs")
    for src, items in sorted(by_source.items()):
        print(f"  {src}: {len(items)}")

    corpus = _pairs_to_text(pairs)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    import sentencepiece as _spm
    sp = _spm.SentencePieceProcessor()
    sp.Load(str(spm_path))

    tokens = sp.EncodeAsIds(corpus)
    if len(tokens) < seq_len * 2:
        # corpus is tiny — repeat it to fill at least a few batches
        repeat = max(1, (seq_len * 8) // max(len(tokens), 1))
        tokens = tokens * repeat
        print(f"[Warning] Corpus very small ({len(tokens)//repeat} tokens). "
              f"Repeating x{repeat} for stability.")

    sequences = [tokens[i:i + seq_len]
                 for i in range(0, max(1, len(tokens) - seq_len), seq_len)]
    sequences = [s if len(s) == seq_len else s + [0] * (seq_len - len(s))
                 for s in sequences]
    dataset  = TextDataset([t for s in sequences for t in s], seq_len=seq_len)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = _load_ckpt(ckpt_path)
    state = ckpt.get("model_state_dict", ckpt)
    vocab_size = ckpt.get("vocab_size", len(sp))

    device = torch.device(device_str if device_str else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  |  vocab: {vocab_size}  |  seq_len: {seq_len}")

    model = TinyTransformerLM(vocab_size=vocab_size, d_model=256, nhead=4,
                               num_layers=4, dim_ff=1024).to(device)
    model.load_state_dict(state, strict=False)

    # fine-tune with a low LR so we don't destroy existing knowledge
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\nFine-tuning for {epochs} epoch(s) on {len(pairs)} feedback pairs...\n")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            v = logits.shape[-1]
            y = y.clamp(0, v - 1)
            loss = criterion(logits.view(-1, v), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        avg = epoch_loss / max(steps, 1)
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(model, out_dir, sp, vocab_size,
                     step="feedback", epoch=epochs,
                     ckpt_name=out_path.name)
    print(f"\nSaved fine-tuned model → {out_path}")
    print("Reload with:")
    print(f"  python Original_sin/aetheria_core.py talk --ckpt {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a checkpoint on feedback.jsonl pairs")
    parser.add_argument("--ckpt", required=True,
                        help="Base checkpoint to fine-tune (e.g. models/aetheria_soul.pt)")
    parser.add_argument("--out", default="",
                        help="Output path. Defaults to same as --ckpt (overwrites).")
    parser.add_argument("--feedback", default=str(ROOT / "data" / "feedback.jsonl"),
                        help="Feedback file (default: data/feedback.jsonl)")
    parser.add_argument("--spm", default=str(ROOT / "data" / "spm.model"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate. Keep low (1e-5 to 1e-4) to avoid forgetting.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path    = Path(args.ckpt)
    feedback_path = Path(args.feedback)
    spm_path     = Path(args.spm)
    out_path     = Path(args.out) if args.out else ckpt_path

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not spm_path.exists():
        print(f"SPM model not found: {spm_path}")
        sys.exit(1)

    finetune(ckpt_path, feedback_path, out_path, spm_path,
             epochs=args.epochs, lr=args.lr,
             batch_size=args.batch_size, seq_len=args.seq_len,
             device_str=args.device)


if __name__ == "__main__":
    main()
