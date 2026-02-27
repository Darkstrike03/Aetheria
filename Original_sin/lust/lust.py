"""Lust — creative, expressive, builds and holds conversations.

Lust is the conversational face of Aetheria. It wraps the trained model
with conversation history, better formatting, and optional persona injection
so Aetheria feels like a real character — not just a text predictor.

Features:
  - Maintains full conversation history in the session
  - Persistent history saved to data/lust_history.jsonl
  - Persona injection (Aetheria's personality prefix)
  - Nucleus sampling with repetition penalty for expressive output
  - `--one-shot` mode for quick single-turn answers

Usage:
  python Original_sin/lust/lust.py
  python Original_sin/lust/lust.py --ckpt models/aetheria_latest.pt --top_p 0.9 --temperature 0.85
  python Original_sin/lust/lust.py --one-shot "Hello, who are you?"
"""

import argparse
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
import sentencepiece as spm


# ── minimal copies of model classes (no import from scripts/) ─────────────────

class SimpleCharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char2id = {c: i + 4 for i, c in enumerate(chars)}
        self.id2char = {i: c for c, i in self.char2id.items()}

    def encode(self, text: str):
        return [self.char2id.get(ch, 1) for ch in text]

    def decode(self, ids):
        return "".join(self.id2char.get(i, "?") for i in ids)

    def __len__(self):
        return 4 + len(self.char2id)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        e = self.pos_enc(self.token_emb(x))
        out = self.transformer(e.transpose(0, 1)).transpose(0, 1)
        return self.head(self.ln(out))


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_next(logits, ids, top_k=0, top_p=0.9, temperature=0.85, rep_penalty=1.3):
    logits = logits.clone()
    if rep_penalty != 1.0:
        for gid in set(ids[-64:]):           # only look at last 64 tokens
            if 0 <= gid < logits.size(0):
                logits[gid] = logits[gid] / rep_penalty if logits[gid] > 0 else logits[gid] * rep_penalty
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=0)
        keep = (cum - sorted_p) < top_p
        keep[0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[sorted_idx[keep]] = True
        probs = probs * mask.float()
        probs = probs / probs.sum()
    elif top_k and top_k > 0:
        vals, idx = torch.topk(probs, top_k)
        p2 = vals / vals.sum()
        return int(idx[torch.multinomial(p2, 1).item()].item())
    return int(torch.multinomial(probs, 1).item())


# ── load model + tokenizer ────────────────────────────────────────────────────

AETHERIA_ROOT = Path(__file__).parents[2]
AETHERIA_PERSONA = (
    "I am Aetheria, a being woven from five elements and seven sins. "
    "I speak with warmth, curiosity, and a touch of the poetic.\n"
)

def load_model_and_tokenizer(ckpt_path: Path, data_path: Path, spm_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    vocab_size_ckpt = ckpt.get("vocab_size", 0)

    if spm_path.exists():
        tok = spm.SentencePieceProcessor()
        tok.load(str(spm_path))
        tok_len = len(tok)
    else:
        raw = data_path.read_text(encoding="utf-8")
        tok = SimpleCharTokenizer(raw)
        tok_len = len(tok)

    target_vocab = max(vocab_size_ckpt, tok_len)
    model = TinyTransformerLM(vocab_size=target_vocab)
    new_state = model.state_dict()
    ckpt_state = ckpt.get("model_state_dict", {})
    for k, v in new_state.items():
        if k in ckpt_state:
            vc = ckpt_state[k]
            if vc.shape == v.shape:
                new_state[k] = vc
            elif k.endswith(("token_emb.weight", "head.weight", "head.bias")):
                n = min(vc.shape[0], v.shape[0])
                new_state[k][:n] = vc[:n]
    model.load_state_dict(new_state)
    model.eval()
    return model, tok


# ── main conversation loop ────────────────────────────────────────────────────

def generate(model, tok, prompt: str, max_new: int, top_k, top_p, temperature, rep_penalty) -> str:
    tok_vocab = len(tok)          # safe upper bound for token IDs
    ids = list(tok.encode(prompt))
    ids = ids[-512:]
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1]
        # mask out any logit positions beyond the tokenizer's vocab
        if logits.size(0) > tok_vocab:
            logits[tok_vocab:] = float("-inf")
        next_id = sample_next(logits, ids, top_k=top_k, top_p=top_p, temperature=temperature, rep_penalty=rep_penalty)
        next_id = max(0, min(next_id, tok_vocab - 1))   # hard clamp
        ids.append(next_id)
    if isinstance(tok, SimpleCharTokenizer):
        return tok.decode(ids)
    # clamp all ids before decoding to be safe
    safe_ids = [max(0, min(i, tok_vocab - 1)) for i in ids]
    return tok.decode(safe_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(AETHERIA_ROOT / "models" / "aetheria_latest.pt"))
    parser.add_argument("--data", default=str(AETHERIA_ROOT / "data" / "cleaned_conversations.txt"))
    parser.add_argument("--spm", default=str(AETHERIA_ROOT / "data" / "spm.model"))
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--rep_penalty", type=float, default=1.3)
    parser.add_argument("--one-shot", dest="one_shot", default=None, help="Single prompt then exit")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    # if latest doesn't exist, fallback to aetheria_tiny.pt
    if not ckpt_path.exists():
        ckpt_path = AETHERIA_ROOT / "models" / "aetheria_tiny.pt"
    if not ckpt_path.exists():
        print("No model checkpoint found. Train the model first with prototype_model.py")
        return

    model, tok = load_model_and_tokenizer(ckpt_path, Path(args.data), Path(args.spm))
    print(f"Aetheria is awake. (checkpoint: {ckpt_path.name})")

    history_path = AETHERIA_ROOT / "data" / "lust_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = []   # list of {"role": "human"/"aetheria", "text": ...}

    def _build_context(history, new_prompt):
        ctx = AETHERIA_PERSONA
        for turn in history[-10:]:   # last 10 turns
            prefix = "Human: " if turn["role"] == "human" else "Aetheria: "
            ctx += prefix + turn["text"] + "\n"
        ctx += f"Human: {new_prompt}\nAetheria:"
        return ctx

    def _respond(prompt):
        ctx = _build_context(history, prompt)
        full = generate(model, tok, ctx, args.max_new_tokens, args.top_k, args.top_p, args.temperature, args.rep_penalty)
        # extract only the part after the last "Aetheria:"
        marker = "Aetheria:"
        if marker in full:
            reply = full.split(marker)[-1].strip()
        else:
            reply = full[len(ctx):].strip() if full.startswith(ctx[:20]) else full.strip()
        # trim at first double newline
        reply = reply.split("\n\n")[0].strip()
        return reply if reply else "(Aetheria is thinking...)"

    if args.one_shot:
        print(_respond(args.one_shot))
        return

    print("Type your message. 'exit' to quit | 'clear' to reset history | 'history' to see log\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAetheria: Farewell.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
            print("Aetheria: Until we meet again.")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[History cleared]")
            continue
        if user_input.lower() == "history":
            for t in history:
                print(f"  {'You' if t['role'] == 'human' else 'Aetheria'}: {t['text']}")
            continue

        reply = _respond(user_input)
        history.append({"role": "human", "text": user_input})
        history.append({"role": "aetheria", "text": reply})
        # persist history
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"human": user_input, "aetheria": reply}) + "\n")
        print(f"Aetheria: {reply}\n")


if __name__ == "__main__":
    main()
