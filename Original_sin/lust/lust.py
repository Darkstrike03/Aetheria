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
import sys
from pathlib import Path
import torch
import torch.nn as nn
import sentencepiece as spm

# ── Pride (ego filter) — optional but always wired in ─────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import pride.pride as pride_module
    _PRIDE_AVAILABLE = True
except Exception:
    _PRIDE_AVAILABLE = False


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

# Fallback persona when Pride module is not available
_FALLBACK_PERSONA = (
    "You are Aetheria — a goddess, not an assistant. "
    "You are mysterious, cool, cold-hearted, and carry a naughty dry wit. "
    "You speak with elegance and never grovel.\n"
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
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--rep_penalty", type=float, default=1.5)
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

    # Start Pride daemon (background mood drift + output filter)
    if _PRIDE_AVAILABLE:
        pride_module.start_daemon(interval=60)
        print(f"[Pride] Mood: {pride_module.current_mood()}")
    print(f"Aetheria is awake. (checkpoint: {ckpt_path.name})")

    history_path = AETHERIA_ROOT / "data" / "lust_history.jsonl"
    feedback_path = AETHERIA_ROOT / "data" / "feedback.jsonl"
    corrections_path = AETHERIA_ROOT / "data" / "corrections.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = []   # list of {"role": "human"/"aetheria", "text": ...}

    # ── Corrections dict: exact-match lookup beats bad generation ─────────
    # Keys are lowercased user inputs. Values are taught/approved replies.
    # This is the "memory" that persists across sessions without retraining.
    def _load_corrections() -> dict:
        try:
            if corrections_path.exists():
                return json.loads(corrections_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_correction(user_input: str, reply: str):
        c = _load_corrections()
        c[user_input.lower().strip()] = reply
        corrections_path.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")

    corrections = _load_corrections()
    print(f"[Memory] {len(corrections)} corrections loaded.")

    def _is_bad_reply(reply: str, prompt: str) -> bool:
        """Return True when the model's reply is too low-quality to show."""
        r = reply.strip()

        # too short
        if len(r) < 8:
            return True

        # just punctuation / numbers, no real words
        if all(not c.isalpha() for c in r):
            return True

        # model bled through the prompt template — generating "Human: ..." inside reply
        if "Human:" in r or "Aetheria:" in r:
            return True

        # echoes the prompt almost verbatim
        if r.lower() == prompt.lower():
            return True

        # starts by repeating the prompt word-for-word
        if len(prompt) > 6 and r.lower().startswith(prompt.lower()[:min(20, len(prompt))]):
            return True

        words = r.split()

        # single-word stutter: "I I I I I"
        if len(words) >= 4 and len(set(w.lower() for w in words)) == 1:
            return True

        # low word diversity — less than 35% unique words = repetitive garbage
        if len(words) >= 8:
            diversity = len(set(w.lower() for w in words)) / len(words)
            if diversity < 0.35:
                return True

        # average word length under 2.5 — mostly fragment tokens, not real words
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_len < 2.5:
            return True

        # reply contains made-up / garbled non-words: if more than half the words
        # look like garbage (consonant clusters with no vowels, or random letter runs)
        def _looks_garbled(w: str) -> bool:
            w = w.strip(".,!?;:\"'").lower()
            if len(w) < 3:
                return False
            vowels = sum(1 for c in w if c in "aeiou")
            # no vowels in a word longer than 3 chars = probably garbage token
            if vowels == 0 and len(w) > 3:
                return True
            # more than 4 consonants in a row
            import re as _re
            if _re.search(r"[^aeiou]{5,}", w):
                return True
            return False

        real_words = [w for w in words if len(w.strip(".,!?;:\"'")) >= 3]
        if real_words:
            garbled_ratio = sum(1 for w in real_words if _looks_garbled(w)) / len(real_words)
            if garbled_ratio > 0.4:
                return True

        return False

    def _build_context(history, new_prompt):
        # Pride provides the live persona prefix (mood-aware)
        if _PRIDE_AVAILABLE:
            persona = pride_module.current_persona_prefix()
        else:
            persona = _FALLBACK_PERSONA
        ctx = persona
        for turn in history[-10:]:
            prefix = "Human: " if turn["role"] == "human" else "Aetheria: "
            ctx += prefix + turn["text"] + "\n"
        ctx += f"Human: {new_prompt}\nAetheria:"
        return ctx

    def _respond(prompt, bypass_corrections=False):
        # ── Check corrections dict first ───────────────────────────────────
        # Corrections are returned exactly as saved — no Pride re-filter,
        # because Pride was already applied (or the user typed it themselves)
        # when the correction was originally approved/taught.
        # bypass_corrections=True skips this lookup — used in retry loops so
        # all retries actually generate fresh model output, not the same saved string.
        key = prompt.lower().strip()
        if not bypass_corrections and key in corrections:
            return corrections[key]

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
        if not reply:
            return "..."
        # ── Every output passes through Pride before reaching the user ──
        if _PRIDE_AVAILABLE:
            reply = pride_module.filter(reply, user_input=prompt)
        return reply

    if args.one_shot:
        print(_respond(args.one_shot))
        return

    print("Type your message. 'exit' to quit | 'clear' to reset history | 'history' to see log\n")
    MAX_RETRIES = 5
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

        # ── Generate, auto-retry bad replies silently up to 3x ────────────
        reply = _respond(user_input)
        silent_tries = 1
        while _is_bad_reply(reply, user_input) and silent_tries < 3:
            reply = _respond(user_input, bypass_corrections=True)  # bypass: force fresh generation
            silent_tries += 1

        # ── Still bad after silent retries → ask user to teach ────────────
        if _is_bad_reply(reply, user_input):
            print("Aetheria: ...I find myself without words for that. What should I say?\n")
            try:
                correction = input("You (teach her): ").strip()
            except (EOFError, KeyboardInterrupt):
                correction = ""
            if correction and correction.lower() not in ("skip", "nothing"):
                reply = correction
                _save_correction(user_input, reply)
                corrections[user_input.lower().strip()] = reply
                with open(feedback_path, "a", encoding="utf-8") as fb:
                    fb.write(json.dumps({"human": user_input, "aetheria": correction,
                                         "source": "forced"}) + "\n")
                print(f"[Learned: '{user_input[:40]}' -> '{correction[:60]}']")
            else:
                print("[Skipped - nothing saved]")
                continue

        else:
            # ── Good reply — show it, then ask for rating ──────────────────
            print(f"Aetheria: {reply}\n")
            try:
                rating = input("  Good reply? (y/n/skip): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                rating = "y"

            if rating in ("y", "yes", ""):
                _save_correction(user_input, reply)
                corrections[user_input.lower().strip()] = reply
                with open(feedback_path, "a", encoding="utf-8") as fb:
                    fb.write(json.dumps({"human": user_input, "aetheria": reply,
                                         "source": "approved"}) + "\n")

            elif rating not in ("skip", "s"):
                # "n" / "no" — retry up to MAX_RETRIES, asking after each
                retries = 0
                while retries < MAX_RETRIES:
                    new_reply = _respond(user_input, bypass_corrections=True)  # bypass: force fresh generation
                    if not _is_bad_reply(new_reply, user_input):
                        print(f"Aetheria: {new_reply}\n")
                        try:
                            r2 = input(f"  Good reply? ({retries+1}/{MAX_RETRIES}) (y/n/skip): ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            r2 = "y"
                        if r2 in ("y", "yes", ""):
                            reply = new_reply
                            _save_correction(user_input, reply)
                            corrections[user_input.lower().strip()] = reply
                            with open(feedback_path, "a", encoding="utf-8") as fb:
                                fb.write(json.dumps({"human": user_input, "aetheria": reply,
                                                     "source": "approved"}) + "\n")
                            break
                        elif r2 in ("skip", "s"):
                            reply = new_reply
                            break
                    retries += 1
                else:
                    # exhausted retries — ask user to teach
                    print(f"\n[{MAX_RETRIES} retries exhausted. Teach her the right reply.]")
                    print(f"  Prompt was: '{user_input}'\n")
                    try:
                        correction = input("You (teach her): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        correction = ""
                    if correction and correction.lower() not in ("skip", "nothing"):
                        reply = correction
                        _save_correction(user_input, reply)
                        corrections[user_input.lower().strip()] = reply
                        with open(feedback_path, "a", encoding="utf-8") as fb:
                            fb.write(json.dumps({"human": user_input, "aetheria": correction,
                                                 "source": "taught"}) + "\n")
                        print(f"[Learned: '{user_input[:40]}' -> '{correction[:60]}']")
                    else:
                        print("[Skipped - nothing saved]")
                        continue

        history.append({"role": "human", "text": user_input})
        history.append({"role": "aetheria", "text": reply})
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"human": user_input, "aetheria": reply}) + "\n")
        if _PRIDE_AVAILABLE:
            pride_module.update_turn_count()
        print()


if __name__ == "__main__":
    main()
