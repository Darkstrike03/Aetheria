"""merge_models.py — combine two Aetheria checkpoints by weight averaging.

Weight averaging ("model soup") produces a model that blends the knowledge
from both parents. Works best when both checkpoints share the same architecture
(same d_model, nhead, num_layers — which all Aetheria checkpoints do).

Usage:
  python scripts/merge_models.py --a models/aetheria_envy_trained.pt --b models/aetheria_colab.pt
  python scripts/merge_models.py --a models/aetheria_envy_trained.pt --b models/aetheria_colab.pt --weight_a 0.4
  python scripts/merge_models.py --a models/aetheria_envy_trained.pt --b models/aetheria_colab.pt --out models/aetheria_merged.pt

Options:
  --weight_a   How much of model A to use (0.0–1.0). Default 0.5 = equal blend.
               Set lower (e.g. 0.3) to favour the Colab model which has more data.
"""

import argparse
from pathlib import Path
import torch


def merge(path_a: Path, path_b: Path, out_path: Path, weight_a: float = 0.5):
    weight_b = 1.0 - weight_a

    print(f"Loading A: {path_a.name}")
    ckpt_a = torch.load(str(path_a), map_location="cpu")

    print(f"Loading B: {path_b.name}")
    ckpt_b = torch.load(str(path_b), map_location="cpu")

    state_a = ckpt_a.get("model_state_dict", ckpt_a)
    state_b = ckpt_b.get("model_state_dict", ckpt_b)

    # Sanity check: warn about shape mismatches
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        print(f"  WARNING: keys only in A (will keep A values): {only_a}")
    if only_b:
        print(f"  WARNING: keys only in B (will keep B values): {only_b}")

    merged_state = {}
    for key in keys_a | keys_b:
        if key in state_a and key in state_b:
            ta, tb = state_a[key], state_b[key]
            if ta.shape == tb.shape and ta.is_floating_point():
                merged_state[key] = weight_a * ta + weight_b * tb
            elif ta.shape == tb.shape:
                # integer tensors (e.g. step counters) — keep the larger
                merged_state[key] = torch.max(ta, tb)
            else:
                # shape mismatch — keep the larger vocab model's tensor
                print(f"  Shape mismatch on '{key}': A={tuple(ta.shape)} B={tuple(tb.shape)} — keeping larger")
                merged_state[key] = ta if ta.numel() >= tb.numel() else tb
        elif key in state_a:
            merged_state[key] = state_a[key]
        else:
            merged_state[key] = state_b[key]

    # Build output checkpoint — carry metadata from whichever has more vocab
    vocab_a = ckpt_a.get("vocab_size", 0)
    vocab_b = ckpt_b.get("vocab_size", 0)
    base = ckpt_a if vocab_a >= vocab_b else ckpt_b
    out_ckpt = {k: v for k, v in base.items() if k != "model_state_dict"}
    out_ckpt["model_state_dict"] = merged_state
    out_ckpt["merged_from"] = [str(path_a), str(path_b)]
    out_ckpt["merge_weights"] = [weight_a, weight_b]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(out_path))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nMerged checkpoint saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"Blend: {weight_a:.0%} {path_a.name}  +  {weight_b:.0%} {path_b.name}")


def main():
    parser = argparse.ArgumentParser(description="Merge two Aetheria .pt checkpoints by weight averaging")
    parser.add_argument("--a", required=True, help="Path to first checkpoint (e.g. local envy model)")
    parser.add_argument("--b", required=True, help="Path to second checkpoint (e.g. Colab gluttony model)")
    parser.add_argument("--out", default="models/aetheria_merged.pt", help="Output path")
    parser.add_argument("--weight_a", type=float, default=0.5,
                        help="Weight for model A (0.0–1.0). 0.5 = equal. Lower = favour B (Colab).")
    args = parser.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)
    out_path = Path(args.out)

    if not path_a.exists():
        print(f"Error: {path_a} not found")
        return
    if not path_b.exists():
        print(f"Error: {path_b} not found")
        return

    merge(path_a, path_b, out_path, args.weight_a)
    print(f"\nTo talk with the merged model:")
    print(f"  python Original_sin/aetheria_core.py talk --ckpt {out_path}")


if __name__ == "__main__":
    main()
