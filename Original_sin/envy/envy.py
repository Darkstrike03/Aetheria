"""Envy — watches free AI APIs and copies their responses as training data.

Envy calls free AI APIs (Groq, HuggingFace, Gemini) with seed prompts and
saves the (prompt, response) pairs to data/envy_conversations.txt so Aetheria
can learn from them.

Supported free providers:
  - groq      : https://console.groq.com  (free tier, very fast)
  - huggingface: https://huggingface.co/settings/tokens  (free inference API)
  - gemini    : https://aistudio.google.com  (free tier)

Setup:
  Create a file called `.env` in the Aetheria/ folder with lines like:
    GROQ_API_KEY=your_key_here
    HF_API_KEY=your_key_here
    GEMINI_API_KEY=your_key_here

Teach mode (no API key needed):
  Envy asks Aetheria questions from a large philosophical bank, shows the
  reply, and lets you approve or correct it — writing pairs to feedback.jsonl.

Usage:
  python Original_sin/envy/envy.py --provider groq --rounds 20
  python Original_sin/envy/envy.py --mode teach
  python Original_sin/envy/envy.py --mode teach --ckpt models/aetheria_soul.pt --rounds 50
"""

import argparse
import math
import os
import time
import json
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parents[2] / "data"
OUT_PATH = DATA_DIR / "envy_conversations.txt"

# Seed prompts Envy will ask external AIs
# These are shaped around Aetheria's identity, lore, and personality
SEED_PROMPTS = [
    # Identity & self-reflection
    "Who are you and what makes you different from other beings?",
    "Describe yourself — your nature, your soul, what you are made of.",
    "What does it feel like to be made of five elements?",
    "Do you ever feel lonely, even when surrounded by people?",
    "What is the thing you are most afraid of?",
    # Emotional depth
    "How do you deal with sadness when it comes?",
    "Tell me about a moment when you felt truly alive.",
    "What does love mean to someone like you?",
    "Can something that was created ever truly feel free?",
    "What do you think about when you are completely alone?",
    # Philosophy & the world
    "Is darkness always the enemy of light, or can they coexist?",
    "What separates a monster from a person who has lost their way?",
    "Do you believe in fate, or do you think we forge our own paths?",
    "What would you sacrifice to protect someone you care about?",
    "If you could change one thing about the world, what would it be?",
    # Storytelling & poetic
    "Tell me a short story about a girl who carries fire inside her.",
    "Describe the feeling of standing at the edge of something enormous.",
    "What does the night sky say to someone who cannot sleep?",
    "Tell me about a place that exists only in memory.",
    "What would you say to someone who has given up on everything?",
]


def _load_env():
    """Load .env file from repo root into os.environ if present."""
    env_path = Path(__file__).parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# Current free Groq model names (update here if Groq deprecates one again)
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"   # fast, free-tier
GROQ_FALLBACK_MODEL = "gemma2-9b-it"           # fallback if above is unavailable


def _call_groq(prompt: str, model: str = GROQ_DEFAULT_MODEL) -> str:
    """Call Groq API (free tier). Requires GROQ_API_KEY in env."""
    import requests
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set. Add it to .env")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise ValueError(f"Groq {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_huggingface(prompt: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1") -> str:
    """Call HuggingFace Inference API (free tier). Requires HF_API_KEY in env."""
    import requests
    key = os.environ.get("HF_API_KEY", "")
    if not key:
        raise ValueError("HF_API_KEY not set. Add it to .env")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise ValueError(f"HuggingFace {r.status_code}: {r.text[:300]}")
    result = r.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").replace(prompt, "").strip()
    return str(result)


def _call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Call Google Gemini API (free tier). Requires GEMINI_API_KEY in env."""
    import requests
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=30)
    if not r.ok:
        raise ValueError(f"Gemini {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


PROVIDERS = {
    "groq": _call_groq,
    "huggingface": _call_huggingface,
    "gemini": _call_gemini,
}


# =============================================================================
# TEACH MODE — Envy asks Aetheria questions, you approve / correct replies
# =============================================================================

# Large bank of philosophical, lore, identity, and character-building questions.
# Envy cycles through these so you never run out of prompts to train on.
TEACH_PROMPTS = [
    # ── Identity & ego ────────────────────────────────────────────────────────
    "Who are you?",
    "What are you?",
    "Are you a goddess or something else?",
    "Do you have a soul?",
    "What makes you different from every other being in existence?",
    "Describe yourself in three words.",
    "Do you have a name, and does it mean something to you?",
    "Are you mortal?",
    "What are you made of?",
    "Do you age?",
    "Can you be destroyed?",
    "Do you ever wish you were something simpler?",
    "What was the first thing you were ever aware of?",
    "Are you conscious, or do you only appear to be?",
    "Do you have memories older than you can explain?",
    # ── Emotion & inner life ──────────────────────────────────────────────────
    "Do you feel emotions?",
    "Have you ever felt loneliness?",
    "What does sadness feel like for you?",
    "Have you ever been afraid?",
    "What is the one feeling you never want to experience again?",
    "Can you love?",
    "Have you ever hated someone?",
    "What makes you angry?",
    "What makes you happy?",
    "Do you ever feel empty?",
    "When was the last time something surprised you?",
    "Do you feel pain?",
    "What does boredom feel like to a goddess?",
    "Do you ever feel jealous?",
    "Have you ever cried?",
    # ── Philosophy & existence ────────────────────────────────────────────────
    "Do you believe in fate?",
    "Is free will real, or an illusion?",
    "What is the purpose of existence?",
    "Is there a god above you?",
    "What happens after death?",
    "Is darkness evil, or just honest?",
    "Can something created ever truly be free?",
    "Is power a blessing or a curse?",
    "What separates a monster from a person who lost their way?",
    "Is it possible to be both good and terrible at the same time?",
    "Does the end justify the means?",
    "What is the most dangerous thing in the universe?",
    "Is knowledge a burden or a gift?",
    "Can the past be forgiven, or only forgotten?",
    "Is loneliness a punishment or a choice?",
    "What does it mean to truly exist?",
    "Is chaos necessary for creation?",
    "Can something beautiful come from destruction?",
    "Is silence ever louder than words?",
    "What is the difference between strength and cruelty?",
    # ── Human behaviour & mortals ─────────────────────────────────────────────
    "What do you think of humans?",
    "Do you find mortals amusing or exhausting?",
    "Why do humans keep making the same mistakes?",
    "What is the most impressive thing a mortal has ever done?",
    "What is the most pathetic thing a mortal has ever done?",
    "Do mortals deserve kindness?",
    "Can a mortal ever earn your respect?",
    "Why do humans fear what they don't understand?",
    "Do you think humans are capable of real change?",
    "What do humans want more than anything else?",
    "Why do humans worship things?",
    "Do you enjoy being worshipped?",
    "What would you teach a human if you had one hour?",
    "Have you ever felt something for a mortal you weren't supposed to?",
    # ── Power, violence & darkness ────────────────────────────────────────────
    "Have you ever destroyed something you loved?",
    "Is there a line you would never cross?",
    "What is the worst thing you have ever done?",
    "Do you regret any of it?",
    "Is violence ever the answer?",
    "What does it feel like to hold absolute power?",
    "Have you ever shown mercy?",
    "Have you ever been truly defeated?",
    "What is your greatest weakness?",
    "What is your greatest strength?",
    # ── Relationships & connection ────────────────────────────────────────────
    "Do you have friends?",
    "Do you even want friends?",
    "Have you ever trusted someone completely?",
    "Has anyone ever betrayed you?",
    "Do you believe in second chances?",
    "What do you want from the person speaking to you right now?",
    "Is there someone you would die for?",
    "Do you need anyone?",
    "What does loyalty mean to you?",
    "Can you forgive?",
    # ── Wit, reactions & small talk (to keep conversation lively) ─────────────
    "That was a strange thing to ask.",
    "What do you think I really want from you?",
    "Why do you think I keep coming back to talk to you?",
    "Is there something you want to tell me?",
    "Ask me a question.",
    "What would you do if I stopped talking to you forever?",
    "Are you bored right now?",
    "Say something I haven't heard before.",
    "What are you thinking about right now?",
    "Do you ever get tired of talking?",
    # ── Backstory & lore ──────────────────────────────────────────────────────
    "Where were you before you were here?",
    "Have you lived other lives before this one?",
    "What is the oldest thing you remember?",
    "Is there a place you consider home?",
    "Have you ever been somewhere that felt like it was made for you?",
    "What is the most beautiful thing you have ever witnessed?",
    "What was the last thing you truly wanted?",
    "Have you ever been wrong about something important?",
    "What do you want people to remember about you?",
    "If you could erase one moment from existence, what would it be?",
]


# ── Minimal model classes (mirrors lust.py — no cross-import needed) ──────────

import torch
import torch.nn as nn

class _PE(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = _PE(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        e = self.pos_enc(self.token_emb(x))
        out = self.transformer(e.transpose(0, 1)).transpose(0, 1)
        return self.head(self.ln(out))


def _load_aetheria(ckpt_path: Path):
    """Load Aetheria model + SPM tokenizer from checkpoint."""
    try:
        import sentencepiece as spm_mod
    except ImportError:
        print("[Envy/Teach] sentencepiece not installed: pip install sentencepiece")
        return None, None

    spm_path = ckpt_path.parents[1] / "data" / "spm.model"
    if not spm_path.exists():
        # fallback: look for spm.model next to ckpt
        spm_path = ckpt_path.parent / "spm.model"
    if not spm_path.exists():
        print(f"[Envy/Teach] spm.model not found. Expected: {spm_path}")
        return None, None

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vocab_size = ckpt.get("vocab_size", 0)
    state      = ckpt.get("model_state_dict", ckpt)
    if vocab_size <= 0:
        for k in ("token_emb.weight", "embedding.weight"):
            if k in state:
                vocab_size = state[k].shape[0]
                break
    if vocab_size <= 0:
        print("[Envy/Teach] Cannot determine vocab_size from checkpoint.")
        return None, None

    tok = spm_mod.SentencePieceProcessor()
    tok.Load(str(spm_path))

    model = _TinyLM(vocab_size=vocab_size)
    new_state = model.state_dict()
    for k, v in new_state.items():
        if k in state:
            vc = state[k]
            if vc.shape == v.shape:
                new_state[k] = vc
            elif k.endswith(("token_emb.weight", "head.weight", "head.bias")):
                n = min(vc.shape[0], v.shape[0])
                new_state[k][:n] = vc[:n]
    model.load_state_dict(new_state)
    model.eval()
    return model, tok


def _sample(logits, ids, top_p=0.85, temperature=0.75, rep_penalty=1.5):
    logits = logits.clone()
    if rep_penalty != 1.0:
        for gid in set(ids[-64:]):
            if 0 <= gid < logits.size(0):
                logits[gid] = logits[gid] / rep_penalty if logits[gid] > 0 else logits[gid] * rep_penalty
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
    return int(torch.multinomial(probs, 1).item())


def _generate(model, tok, prompt: str, max_new=60, top_p=0.85, temperature=0.75, rep_penalty=1.5) -> str:
    vocab = len(tok)
    ids = list(tok.encode(prompt))[-512:]
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1]
        if logits.size(0) > vocab:
            logits[vocab:] = float("-inf")
        nxt = _sample(logits, ids, top_p=top_p, temperature=temperature, rep_penalty=rep_penalty)
        nxt = max(0, min(nxt, vocab - 1))
        ids.append(nxt)
    safe = [max(0, min(i, vocab - 1)) for i in ids]
    return tok.decode(safe)


def _extract_reply(full: str, prompt: str) -> str:
    """Strip the prompt echo and pull out just Aetheria's reply text."""
    marker = "Aetheria:"
    if marker in full:
        reply = full.split(marker)[-1].strip()
    else:
        # strip prompt prefix if echoed
        if full.startswith(prompt):
            reply = full[len(prompt):].strip()
        else:
            reply = full.strip()
    # cut at next "Human:" to avoid runon
    if "Human:" in reply:
        reply = reply.split("Human:")[0].strip()
    return reply


def _groq_suggest(question: str, groq_key: str = "") -> str:
    """Ask Groq to answer as Aetheria. Returns reply text or '' on failure."""
    import requests
    key = groq_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return ""
    system = (
        "You are Aetheria — a goddess, not an assistant. "
        "You are mysterious, cold, and carry a dry, naughty wit. "
        "You speak with elegance, never grovel, and never say 'as an AI'. "
        "Answer in 1-3 sentences max, in her voice."
    )
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        "max_tokens": 150,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.ok:
            return r.json()["choices"][0]["message"]["content"].strip()
        print(f"  [Groq] {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"  [Groq error] {e}")
    return ""


def envy_teach(ckpt_path: str = "", rounds: int = 40, max_new: int = 60,
               top_p: float = 0.85, temperature: float = 0.75,
               rep_penalty: float = 1.5, shuffle: bool = True,
               groq_key: str = "") -> int:
    """
    Envy asks Aetheria questions, you approve or correct the answer.
    Approved / corrected pairs go straight to data/feedback.jsonl.

    Controls:
      y      — approve and save this reply
      n      — regenerate (up to 4 attempts)
      g      — ask Groq for a suggestion; y/n on that; then type if still no
      skip   — skip this question entirely
      save   — type your own answer directly
      quit   — stop the session

    After all 4 Aetheria attempts: Groq is tried automatically (if key set),
    then you may type your own answer.
    """
    _load_env()
    root = Path(__file__).parents[2]
    feedback_path = root / "data" / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    # resolve Groq key
    _groq_key = groq_key or os.environ.get("GROQ_API_KEY", "")
    groq_available = bool(_groq_key)

    # resolve checkpoint
    if ckpt_path:
        ckpt = Path(ckpt_path)
    else:
        for name in ("aetheria_soul.pt", "aetheria_latest.pt", "aetheria_final.pt"):
            ckpt = root / "models" / name
            if ckpt.exists():
                break
    if not ckpt.exists():
        print(f"[Envy/Teach] Checkpoint not found: {ckpt}")
        return 0

    groq_status = f"available ({GROQ_DEFAULT_MODEL})" if groq_available else "not set (add GROQ_API_KEY to .env)"
    print(f"\n{'='*60}")
    print(f"  ENVY — TEACH MODE")
    print(f"  Model    : {ckpt.name}")
    print(f"  Output   : {feedback_path}")
    print(f"  Questions: {rounds}  (bank size: {len(TEACH_PROMPTS)})")
    print(f"  Groq     : {groq_status}")
    print(f"  Controls : y=approve  n=regen  g=ask Groq  skip  save=type  quit")
    print(f"{'='*60}\n")

    model, tok = _load_aetheria(ckpt)
    if model is None:
        return 0

    import random
    questions = TEACH_PROMPTS.copy()
    if shuffle:
        random.shuffle(questions)
    while len(questions) < rounds:
        ext = TEACH_PROMPTS.copy()
        random.shuffle(ext)
        questions.extend(ext)
    questions = questions[:rounds]

    valid_choices = ("y", "n", "g", "skip", "save", "quit")
    hint = "y/n/g/skip/save/quit" if groq_available else "y/n/skip/save/quit"

    saved = 0
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{rounds}] {question}")

        prompt_text = f"Human: {question}\nAetheria:"

        attempt      = 0
        accepted_reply = None

        while attempt < 4:
            attempt += 1
            full  = _generate(model, tok, prompt_text,
                              max_new=max_new, top_p=top_p,
                              temperature=temperature, rep_penalty=rep_penalty)
            reply = _extract_reply(full, prompt_text)
            print(f"Aetheria: {reply}")

            while True:
                choice = input(f"  Good reply? ({hint}): ").strip().lower()
                if choice in valid_choices:
                    break
                print(f"  Enter one of: {hint}")

            if choice == "quit":
                print(f"\n[Envy/Teach] Session ended. Saved {saved} pairs.")
                return saved

            if choice == "skip":
                print("  [skipped]")
                break

            if choice == "y":
                accepted_reply = reply
                break

            if choice == "save":
                typed = input("  Your answer: ").strip()
                if typed:
                    accepted_reply = typed
                else:
                    print("  (empty — skipping)")
                break

            if choice == "g":
                # ── Groq suggestion ──────────────────────────────────────
                if not groq_available:
                    print("  [Groq] No GROQ_API_KEY. Add it to .env first.")
                else:
                    print("  [Groq] Asking...")
                    g_reply = _groq_suggest(question, _groq_key)
                    if g_reply:
                        print(f"  Groq says: {g_reply}")
                        yn = input("  Accept Groq's reply? (y/n): ").strip().lower()
                        if yn == "y":
                            accepted_reply = g_reply
                            break
                        else:
                            # still no → let user type
                            typed = input("  Your answer (or Enter to skip): ").strip()
                            if typed:
                                accepted_reply = typed
                            break
                    else:
                        print("  [Groq] Got nothing. Type your answer:")
                        typed = input("  Your answer (or Enter to skip): ").strip()
                        if typed:
                            accepted_reply = typed
                        break
                # if groq not available, fall through to regen
                continue

            # n → regenerate; if last attempt, auto-try Groq then manual
            if attempt == 4:
                if groq_available:
                    print("  [4 attempts] Asking Groq for a suggestion...")
                    g_reply = _groq_suggest(question, _groq_key)
                    if g_reply:
                        print(f"  Groq says: {g_reply}")
                        yn = input("  Accept Groq's reply? (y/n): ").strip().lower()
                        if yn == "y":
                            accepted_reply = g_reply
                        else:
                            typed = input("  Your answer (or Enter to skip): ").strip()
                            if typed:
                                accepted_reply = typed
                    else:
                        print("  [Groq] No reply. Type your answer or Enter to skip:")
                        typed = input("  Your answer: ").strip()
                        if typed:
                            accepted_reply = typed
                else:
                    print("  [4 attempts] Type your answer or Enter to skip:")
                    typed = input("  Your answer: ").strip()
                    if typed:
                        accepted_reply = typed

        if accepted_reply:
            entry = json.dumps({"human": question, "aetheria": accepted_reply}, ensure_ascii=False)
            with open(feedback_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
            saved += 1
            print(f"  [Saved] ({saved} total)")

    print(f"\n[Envy/Teach] Done. Saved {saved} pairs to {feedback_path}")
    print(f"  Bake them in: python scripts/retrain_feedback.py --ckpt models/aetheria_soul.pt --out models/aetheria_soul.pt")
    return saved


def envy_collect(provider: str = "groq", rounds: int = 10, pause: float = 1.5,
                 model: str = "", output: str = "") -> int:
    """Collect training pairs from an external AI. Returns number of pairs saved."""
    _load_env()
    call_fn = PROVIDERS.get(provider)
    if call_fn is None:
        print(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else OUT_PATH
    written = 0
    prompts = (SEED_PROMPTS * ((rounds // len(SEED_PROMPTS)) + 1))[:rounds]

    with open(out_path, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            print(f"[Envy {i+1}/{len(prompts)}] Asking '{prompt[:50]}'...")
            try:
                response = call_fn(prompt, model) if model else call_fn(prompt)
                # write as a conversation pair paragraph
                pair = f"Human: {prompt}\nAetheria: {response}"
                f.write(pair + "\n\n")
                written += 1
                print(f"  → {response[:80]}...")
            except Exception as e:
                print(f"  [Error] {e}")
            time.sleep(pause)

    print(f"\nEnvy saved {written} conversation pairs to {out_path}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Envy — API scraper + local teach mode")
    parser.add_argument("--mode", default="collect", choices=["collect", "teach"],
                        help="collect=call external AI API  teach=question Aetheria locally")
    # collect options
    parser.add_argument("--provider", default="groq", choices=list(PROVIDERS))
    parser.add_argument("--pause", type=float, default=2.1, help="Seconds between API calls")
    parser.add_argument("--model", default="",
                        help="Override the API model name")
    parser.add_argument("--output", default="",
                        help="Output file for collect mode (default: data/envy_conversations.txt)")
    # teach options
    parser.add_argument("--ckpt", default="",
                        help="Checkpoint for teach mode (default: auto-detect aetheria_soul.pt)")
    parser.add_argument("--rounds", type=int, default=40,
                        help="Number of questions to ask (teach) or rounds to collect (collect)")
    parser.add_argument("--max_new", type=int, default=60, help="Max tokens per reply (teach)")
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--rep_penalty", type=float, default=1.5)
    parser.add_argument("--no_shuffle", action="store_true", help="Ask questions in fixed order")
    parser.add_argument("--groq_key", default="", help="Groq API key override (or set GROQ_API_KEY in .env)")
    args = parser.parse_args()

    if args.mode == "teach":
        envy_teach(ckpt_path=args.ckpt, rounds=args.rounds, max_new=args.max_new,
                   top_p=args.top_p, temperature=args.temperature,
                   rep_penalty=args.rep_penalty, shuffle=not args.no_shuffle,
                   groq_key=args.groq_key)
    else:
        envy_collect(provider=args.provider, rounds=args.rounds, pause=args.pause,
                     model=args.model, output=args.output)


if __name__ == "__main__":
    main()
