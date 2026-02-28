"""merge_all.py — score every model in models/ and merge the best into aetheria_U.pt

How it works:
  1. Loads every .pt checkpoint in models/
  2. Scores each by perplexity on data/persona_seed_clean.txt
     (lower perplexity = model predicts Aetheria's voice better = higher weight)
  3. Optionally skips models above a perplexity threshold (--max_perplexity)
  4. Weighted-averages the survivors into models/aetheria_U.pt

Usage:
  python scripts/merge_all.py
  python scripts/merge_all.py --max_perplexity 200  # skip very bad models
  python scripts/merge_all.py --top 5               # only use top 5 models
  python scripts/merge_all.py --eval_data data/persona_seed_clean.txt
  python scripts/merge_all.py --out models/aetheria_U.pt
  python scripts/merge_all.py --dry_run             # just print scores, don't merge
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# ── Import model architecture from prototype_model ───────────────────────────
from prototype_model import TinyTransformerLM   # noqa: E402


def _load_ckpt(path: Path):
    """Load a checkpoint dict. Returns (state_dict, vocab_size, meta)."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("vocab_size", 0), ckpt
    # bare state_dict
    return ckpt, 0, {}


def _infer_vocab_size(state: dict) -> int:
    """Infer vocab size from embedding weight shape."""
    for key in ("embedding.weight", "tok_emb.weight", "embed.weight"):
        if key in state:
            return state[key].shape[0]
    return 0


def _check_shape_compatible(state: dict, vocab_size: int) -> tuple[bool, str]:
    """Check that all weight shapes match TinyTransformerLM(d_model=256, nhead=4, num_layers=4).
    Returns (compatible: bool, reason: str)."""
    expected = {
        "tok_emb.weight":       (vocab_size, 256),
        "blocks.0.attn.in_proj_weight": (768, 256),   # 3*256
        "blocks.0.ff.0.weight": (1024, 256),
        "ln_f.weight":          (256,),
        "head.weight":          (vocab_size, 256),
    }
    # only check keys that exist in state
    for key, shape in expected.items():
        if key in state and tuple(state[key].shape) != shape:
            return False, f"{key}: expected {shape}, got {tuple(state[key].shape)}"
    # check d_model via any embedding key
    for emb_key in ("tok_emb.weight", "embedding.weight", "embed.weight"):
        if emb_key in state:
            d_model = state[emb_key].shape[1]
            if d_model != 256:
                return False, f"d_model mismatch: {emb_key} has d_model={d_model}, expected 256"
            break
    return True, "ok"


def _build_model(state: dict, vocab_size: int) -> TinyTransformerLM:
    if vocab_size <= 0:
        vocab_size = _infer_vocab_size(state)
    if vocab_size <= 0:
        raise ValueError("Cannot determine vocab_size from checkpoint")
    compatible, reason = _check_shape_compatible(state, vocab_size)
    if not compatible:
        raise ValueError(f"Architecture mismatch — {reason}")
    model = TinyTransformerLM(vocab_size=vocab_size, d_model=256, nhead=4,
                              num_layers=4, dim_ff=1024)
    model.load_state_dict(state, strict=True)   # strict=True: fail loudly on mismatch
    model.eval()
    return model


def _load_spm(path: Path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(path))
    return sp


def _perplexity(model: TinyTransformerLM, tokens: list[int],
                seq_len: int = 64, device: str = "cpu") -> float:
    """Compute perplexity of the model on a token list."""
    model = model.to(device)
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for i in range(0, max(1, len(tokens) - seq_len), seq_len):
            chunk = tokens[i: i + seq_len + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]],  dtype=torch.long, device=device)
            logits = model(x)                    # (1, T, V)
            # clamp target ids to vocab
            v = logits.shape[-1]
            y = y.clamp(0, v - 1)
            logits_flat = logits.view(-1, v)
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            total_loss += loss.item()
            total_tokens += y_flat.numel()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def score_all(model_dir: Path, eval_path: Path, spm_path: Path,
              max_ppl: float = float("inf"), top_n: int = 0) -> list[tuple]:
    """
    Returns list of (path, perplexity, weight) sorted best-first,
    filtered to max_ppl, limited to top_n if given.
    """
    pts = sorted(model_dir.glob("*.pt"))
    if not pts:
        print("No .pt files found in", model_dir)
        return []

    # load tokenizer
    sp = _load_spm(spm_path)
    raw = eval_path.read_text(encoding="utf-8", errors="ignore")
    tokens = sp.EncodeAsIds(raw)
    if not tokens:
        print("Warning: eval data produced 0 tokens — check spm model matches your data")
        return []

    print(f"\nScoring {len(pts)} model(s) on {eval_path.name} ({len(tokens)} tokens)...\n")

    results = []
    for pt in pts:
        try:
            state, vocab_size, _meta = _load_ckpt(pt)
            model = _build_model(state, vocab_size)
            ppl = _perplexity(model, tokens)
            status = "✓" if ppl <= max_ppl else "✗ (excluded)"
            print(f"  {pt.name:<35}  ppl = {ppl:>10.2f}  {status}")
            if ppl <= max_ppl:
                results.append((pt, ppl))
        except Exception as e:
            print(f"  {pt.name:<35}  ERROR: {e}")

    if not results:
        print("\nNo models passed the perplexity threshold.")
        return []

    # sort best-first
    results.sort(key=lambda x: x[1])

    if top_n and top_n < len(results):
        print(f"\nKeeping top {top_n} / {len(results)} models.")
        results = results[:top_n]

    # compute softmax weights: w_i = softmax(-log(ppl_i))
    # lower perplexity → higher log-likelihood → higher weight
    log_likes = [-math.log(max(ppl, 1e-9)) for _, ppl in results]
    max_ll = max(log_likes)
    exp_ll = [math.exp(ll - max_ll) for ll in log_likes]   # numerically stable
    total = sum(exp_ll)
    weights = [e / total for e in exp_ll]

    final = [(pt, ppl, w) for (pt, ppl), w in zip(results, weights)]
    return final


def merge_weighted(scored: list[tuple], out_path: Path):
    """Weighted average of all checkpoints in scored list."""
    print("\n── Merging ─────────────────────────────────────────────────────────")
    for pt, ppl, w in scored:
        print(f"  {w:5.1%}  {pt.name}  (ppl={ppl:.1f})")

    # load all states
    states = []
    for pt, _ppl, _w in scored:
        state, vocab_size, _meta = _load_ckpt(pt)
        states.append((state, vocab_size))

    # union of all keys
    all_keys = set()
    for state, _ in states:
        all_keys.update(state.keys())

    weights = [w for _, _, w in scored]
    merged = {}
    for key in all_keys:
        tensors = [(state[key], weights[i])
                   for i, (state, _) in enumerate(states)
                   if key in state]
        if not tensors:
            continue

        t0, _ = tensors[0]
        if not t0.is_floating_point():
            merged[key] = t0   # keep from best model
            continue

        # check all same shape — if not, keep largest
        shapes = [t.shape for t, _ in tensors]
        if len(set(shapes)) > 1:
            biggest = max(tensors, key=lambda x: x[0].numel())
            merged[key] = biggest[0]
            continue

        # weighted sum
        acc = torch.zeros_like(t0)
        total_w = sum(w for _, w in tensors)
        for t, w in tensors:
            acc = acc + (w / total_w) * t
        merged[key] = acc

    # pick metadata from the best (lowest ppl) model
    _, _, best_meta = _load_ckpt(scored[0][0])
    out_ckpt = {k: v for k, v in best_meta.items() if k != "model_state_dict"}
    out_ckpt["model_state_dict"] = merged
    out_ckpt["merged_from"] = [str(pt) for pt, _, _ in scored]
    out_ckpt["merge_weights"] = [round(w, 4) for _, _, w in scored]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved  →  {out_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Score all models and merge the best into aetheria_U.pt")
    parser.add_argument("--model_dir", default=str(ROOT / "models"),
                        help="Directory containing .pt checkpoints")
    parser.add_argument("--eval_data", default=str(ROOT / "data" / "persona_seed_clean.txt"),
                        help="Text file to evaluate perplexity on")
    parser.add_argument("--spm", default=str(ROOT / "data" / "spm.model"),
                        help="SentencePiece model path")
    parser.add_argument("--out", default=str(ROOT / "models" / "aetheria_U.pt"),
                        help="Output merged checkpoint")
    parser.add_argument("--max_perplexity", type=float, default=float("inf"),
                        help="Exclude models with perplexity above this value")
    parser.add_argument("--top", type=int, default=0,
                        help="Only use top N models (0 = use all that pass threshold)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print scores only, do not merge")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    eval_path = Path(args.eval_data)
    spm_path  = Path(args.spm)
    out_path  = Path(args.out)

    if not eval_path.exists():
        # fallback: try persona_seed.txt
        fallback = ROOT / "data" / "persona_seed.txt"
        if fallback.exists():
            eval_path = fallback
            print(f"[Warning] eval_data not found, using {fallback.name}")
        else:
            print(f"[Error] Eval data not found: {eval_path}")
            print("  Run: python scripts/build_persona_seed.py  to generate it")
            sys.exit(1)

    if not spm_path.exists():
        print(f"[Error] SPM model not found: {spm_path}")
        print("  Run: python scripts/train_spm.py  to generate it")
        sys.exit(1)

    scored = score_all(model_dir, eval_path, spm_path,
                       max_ppl=args.max_perplexity, top_n=args.top)

    if not scored:
        sys.exit(1)

    print(f"\n── Rankings ────────────────────────────────────────────────────────")
    for rank, (pt, ppl, w) in enumerate(scored, 1):
        print(f"  #{rank}  {pt.name:<35}  ppl={ppl:>10.2f}  weight={w:.1%}")

    if args.dry_run:
        print("\n[dry_run] Skipping merge.")
        return

    merge_weighted(scored, out_path)
    print("\nDone. Talk to the merged model:")
    print(f"  python Original_sin/aetheria_core.py talk --ckpt {out_path}")


if __name__ == "__main__":
    main()
