"""Gluttony — devours other models and absorbs their knowledge into Aetheria.

Inspired by Rimuru's Predator skill: absorb the target, gain their abilities,
purified by Great Sage (Pride) so the host identity is never corrupted.

Three modes:

  devour   (DEFAULT)
      Absorbs a teacher model: generates responses to seed prompts, then
      immediately fine-tunes the existing Aetheria checkpoint on them.
      Pride persona seed acts as a regularizer every N steps to prevent
      the foreign knowledge from overwriting Aetheria ego.
      Output: models/aetheria_devoured.pt

  generate
      Only generates text pairs from teacher (Envy-adjacent).
      Output: data/gluttony_conversations.txt

  distill
      True KL-divergence distillation. Builds a new student in teacher vocab.
      Output: models/aetheria_distilled.pt

Usage:
  python Original_sin/gluttony/gluttony.py --ckpt models/aetheria_soul.pt
  python Original_sin/gluttony/gluttony.py --mode devour --model dialogpt-medium --ckpt models/aetheria_soul.pt
  python Original_sin/gluttony/gluttony.py --mode generate --model dialogpt-medium --rounds 100
  python Original_sin/gluttony/gluttony.py --mode distill --model dialogpt-medium --epochs 5
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT       = Path(__file__).parents[2]
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUT_PATH   = DATA_DIR / "gluttony_conversations.txt"

sys.path.insert(0, str(ROOT / "scripts"))

# Casual, everyday prompts DialoGPT-medium was trained on (Reddit-style chat).
# Goal: absorb natural conversational flow — greetings, reactions, small talk.
# Save philosophical/deep prompts for later when we have a better teacher.
SEED_PROMPTS = [
    "Hey, how's it going?",
    "What's up?",
    "How are you doing today?",
    "What did you do today?",
    "Did you sleep well?",
    "Are you bored right now?",
    "What do you usually do on weekends?",
    "Do you like music?",
    "What kind of movies do you like?",
    "Have you eaten anything good lately?",
    "Do you have any hobbies?",
    "What's your favourite thing to do when you're free?",
    "Do you prefer staying in or going out?",
    "Do you like talking to people?",
    "What annoys you the most?",
    "What makes you happy?",
    "Do you have a best friend?",
    "What's something funny that happened to you recently?",
    "Do you like animals?",
    "What do you think about when you can't sleep?",
    "Do you ever feel like nobody understands you?",
    "What's the last thing that made you laugh?",
    "What's the most embarrassing thing you've ever done?",
    "Do you like being alone sometimes?",
    "What do you do when you're upset?",
    "Are you the kind of person who talks a lot or listens more?",
    "Do you ever get nervous around people?",
    "What's the nicest thing someone has done for you?",
    "Do you believe in second chances?",
    "What's the weirdest thing you believe in?",
]

KNOWN_MODELS = {
    "dialogpt-small":  "microsoft/DialoGPT-small",
    "dialogpt-medium": "microsoft/DialoGPT-medium",
    "gpt2":            "gpt2",
    "gpt2-medium":     "gpt2-medium",
    "tinyllama":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm":          "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen":            "Qwen/Qwen2.5-0.5B-Instruct",
}

CHAT_MODELS = {"tinyllama", "smollm", "qwen"}


def _wrap_prompt(model_key: str, prompt: str, tokenizer=None) -> str:
    key = model_key.lower()
    if key == "tinyllama":
        return ("<|system|>\nYou are Aetheria — a goddess, not an assistant. "
                "Mysterious, cool, dry wit.</s>\n<|user|>\n" + prompt + "</s>\n<|assistant|>\n")
    if key == "smollm":
        return ("<|im_start|>system\nYou are Aetheria.<|im_end|>\n"
                "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
    if key == "qwen":
        return ("<|im_start|>system\nYou are Aetheria.<|im_end|>\n"
                "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
    # DialoGPT family: append eos_token as turn separator so the model knows
    # the user turn has ended and it should generate the reply turn.
    if "dialogpt" in key or "dialo" in key:
        eos = tokenizer.eos_token if tokenizer is not None else "<|endoftext|>"
        return prompt + eos
    return prompt


def _load_teacher(model_key: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = KNOWN_MODELS.get(model_key, model_key)
    print(f"[Gluttony] Loading teacher: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Gluttony] Teacher ready. ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model, tokenizer


def _teacher_respond(teacher, tokenizer, model_key, prompt, device, max_new_tokens=120):
    wrapped = _wrap_prompt(model_key, prompt, tokenizer=tokenizer)
    inputs = tokenizer(wrapped, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = teacher.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            top_k=50, top_p=0.9, temperature=0.85,
            pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2,
        )
    reply = tokenizer.decode(out[:, input_len:][0], skip_special_tokens=True).strip()
    # Quality gate: reject very short or empty replies (< 3 words)
    if len(reply.split()) < 3:
        return ""
    return reply


# =============================================================================
# MODE 1: DEVOUR  (absorb + purify — Predator + Great Sage)
# =============================================================================

def gluttony_devour(model_key="dialogpt-medium", ckpt_path=None, out_path=None,
                    rounds=30, epochs=3, lr=3e-4, seq_len=64, batch_size=4,
                    pride_weight=0.3, pride_interval=5, device_str=None):
    """
    Absorb a teacher model into existing Aetheria weights.

    Phase 1 ABSORB  - ask teacher every seed prompt, get its responses
    Phase 2 DIGEST  - fine-tune Aetheria checkpoint on absorbed pairs
    Phase 3 PURIFY  - Pride regularization every pride_interval steps
                      keeps Aetheria ego intact (corruption guard)
    """
    try:
        from prototype_model import TinyTransformerLM
        import sentencepiece as _spm_module
    except ImportError as e:
        print(f"[Gluttony] Missing: {e}  ->  pip install sentencepiece transformers")
        return

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*62}")
    print(f"  GLUTTONY — DEVOUR MODE   (Predator + Great Sage)")
    print(f"  Teacher      : {KNOWN_MODELS.get(model_key, model_key)}")
    print(f"  Device       : {device}")
    print(f"  Pride anchor : weight={pride_weight}  every {pride_interval} steps")
    print(f"{'='*62}\n")

    if ckpt_path is None:
        ckpt_path = MODELS_DIR / "aetheria_soul.pt"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[Gluttony] Checkpoint not found: {ckpt_path}")
        return

    spm_path = ROOT / "data" / "spm.model"
    if not spm_path.exists():
        print(f"[Gluttony] SPM model not found: {spm_path} — run train_spm.py first")
        return

    persona_path = ROOT / "data" / "persona_seed.txt"
    if not persona_path.exists():
        print(f"[Gluttony] No persona_seed.txt — run: python scripts/build_persona_seed.py")
        return

    out_path = Path(out_path) if out_path else MODELS_DIR / "aetheria_devoured.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load Aetheria (the host)
    print("[Pride] Loading Aetheria (the host)...")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    vocab_size = ckpt.get("vocab_size", 0)
    if vocab_size <= 0:
        for k in ("embedding.weight", "tok_emb.weight", "embed.weight"):
            if k in state:
                vocab_size = state[k].shape[0]
                break
    if vocab_size <= 0:
        print("[Gluttony] Cannot determine vocab_size.")
        return

    model = TinyTransformerLM(vocab_size=vocab_size, d_model=256, nhead=4,
                               num_layers=4, dim_ff=1024).to(device)
    model.load_state_dict(state, strict=False)

    sp = _spm_module.SentencePieceProcessor()
    sp.Load(str(spm_path))

    # load Pride ego anchor
    print("[Pride] Loading ego anchor (persona_seed.txt)...")
    persona_ids = sp.EncodeAsIds(persona_path.read_text(encoding="utf-8", errors="ignore"))
    while len(persona_ids) < seq_len * batch_size * 4:
        persona_ids = persona_ids * 2
    persona_t = torch.tensor(persona_ids, dtype=torch.long)

    def _persona_batch():
        start = torch.randint(0, max(1, len(persona_t) - seq_len - 1), (batch_size,))
        xs = torch.stack([persona_t[s: s + seq_len] for s in start]).to(device)
        ys = torch.stack([persona_t[s + 1: s + seq_len + 1] for s in start]).to(device)
        return xs, ys

    # Phase 1: ABSORB
    print("\n[Gluttony] Phase 1 — ABSORB ...")
    teacher, t_tok = _load_teacher(model_key, str(device))
    prompts = (SEED_PROMPTS * ((rounds // len(SEED_PROMPTS)) + 1))[:rounds]
    absorbed = []
    for i, p in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {p[:70]}")
        try:
            r = _teacher_respond(teacher, t_tok, model_key, p, str(device))
            if r:
                absorbed.append((p, r))
                print(f"    -> {r[:90]}")
            else:
                print(f"    -> [skipped: response too short or empty]")
        except Exception as e:
            print(f"    -> [Error: {e}]")
    del teacher, t_tok
    if str(device) == "cuda":
        torch.cuda.empty_cache()
    print(f"\n[Gluttony] Absorbed {len(absorbed)} pairs.")
    if not absorbed:
        print("[Gluttony] Nothing absorbed — aborting.")
        return

    # Phase 2: DIGEST
    print("\n[Gluttony] Phase 2 — DIGEST ...")
    abs_ids = []
    for prompt, resp in absorbed:
        abs_ids.extend(sp.EncodeAsIds(f"Human: {prompt}\nAetheria: {resp}"))
        eos = sp.eos_id()
        abs_ids.append(eos if eos > 0 else 1)
    while len(abs_ids) < seq_len * batch_size * 10:
        abs_ids = abs_ids * 2
    abs_t = torch.tensor(abs_ids, dtype=torch.long)

    def _abs_batch():
        start = torch.randint(0, max(1, len(abs_t) - seq_len - 1), (batch_size,))
        xs = torch.stack([abs_t[s: s + seq_len] for s in start]).to(device)
        ys = torch.stack([abs_t[s + 1: s + seq_len + 1] for s in start]).to(device)
        return xs, ys

    # Phase 3: TRAIN with Pride regularization
    print(f"\n[Gluttony] Phase 3 — PURIFY (Pride weight={pride_weight}) ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    steps_per_epoch = max(20, len(abs_ids) // (seq_len * batch_size))
    model.train()

    for epoch in range(epochs):
        t_loss = a_loss = p_loss = steps = 0
        for step in range(steps_per_epoch):
            ax, ay = _abs_batch()
            al = model(ax)
            v  = al.shape[-1]
            ay = ay.clamp(0, v - 1)
            absorb_loss = criterion(al.view(-1, v), ay.view(-1))

            if pride_weight > 0 and step % pride_interval == 0:
                px, py = _persona_batch()
                pl = model(px)
                py = py.clamp(0, v - 1)
                pride_loss = criterion(pl.view(-1, v), py.view(-1))
                loss = (1 - pride_weight) * absorb_loss + pride_weight * pride_loss
                p_loss += pride_loss.item()
            else:
                loss = absorb_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            a_loss += absorb_loss.item()
            steps += 1

        p_steps = max(steps // max(pride_interval, 1), 1)
        print(f"  Epoch {epoch+1}/{epochs}  total={t_loss/max(steps,1):.4f}  "
              f"absorb={a_loss/max(steps,1):.4f}  pride={p_loss/p_steps:.4f}")

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    meta.update({"model_state_dict": model.state_dict(), "vocab_size": vocab_size,
                 "devoured_from": KNOWN_MODELS.get(model_key, model_key),
                 "pride_weight": pride_weight, "absorbed_pairs": len(absorbed)})
    torch.save(meta, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n[Gluttony] Devoured model saved -> {out_path}  ({size_mb:.1f} MB)")
    print(f"  Talk: python Original_sin/aetheria_core.py talk --ckpt {out_path}")


# =============================================================================
# MODE 2: GENERATE  (text pairs only)
# =============================================================================

def gluttony_collect(model_key="dialogpt-medium", rounds=30, out_path=OUT_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, tokenizer = _load_teacher(model_key, device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = (SEED_PROMPTS * ((rounds // len(SEED_PROMPTS)) + 1))[:rounds]
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] {p[:70]}")
            try:
                r = _teacher_respond(teacher, tokenizer, model_key, p, device)
                if r:
                    f.write(f"Human: {p}\nAetheria: {r}\n\n")
                    written += 1
                    print(f"  -> {r[:90]}")
            except Exception as e:
                print(f"  -> [Error: {e}]")
    print(f"\nSaved {written} pairs to {out_path}")
    return written


# =============================================================================
# MODE 3: DISTILL  (KL divergence, new student from scratch — advanced)
# =============================================================================

class _StudentLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(2048, d_model)
        pos = torch.arange(0, 2048).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        e = self.token_emb(x) + self.pe[:, :x.size(1), :]
        return self.head(self.ln(self.transformer(e)))


def gluttony_distill(model_key="dialogpt-medium", epochs=3, seq_len=128,
                     batch_size=4, temperature=3.0, alpha=0.7):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, tokenizer = _load_teacher(model_key, device)
    v = tokenizer.vocab_size or len(tokenizer)
    student = _StudentLM(vocab_size=v).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    kl  = nn.KLDivLoss(reduction="batchmean")
    ids = []
    for p in SEED_PROMPTS:
        ids.extend(tokenizer.encode(_wrap_prompt(model_key, p)))
    while len(ids) < seq_len * batch_size * 20:
        ids = ids * 2
    corp = torch.tensor(ids, dtype=torch.long)
    student.train()
    for epoch in range(epochs):
        idxs = torch.randperm(max(1, len(corp) - seq_len - 1))
        el = es = 0
        bx, by = [], []
        for idx in idxs[:200]:
            x = corp[idx: idx + seq_len]
            y = corp[idx + 1: idx + seq_len + 1]
            if len(x) < seq_len or len(y) < seq_len:
                continue
            bx.append(x); by.append(y)
            if len(bx) < batch_size:
                continue
            BX = torch.stack(bx).to(device); BY = torch.stack(by).to(device)
            bx, by = [], []
            with torch.no_grad():
                tl = teacher(BX).logits.float()
            ts = F.softmax(tl / temperature, dim=-1)
            sl = student(BX).float()
            dl = kl(F.log_softmax(sl / temperature, dim=-1).view(-1, v),
                    ts.view(-1, v)) * temperature ** 2
            hl = F.cross_entropy(sl.view(-1, v), BY.view(-1),
                                  ignore_index=tokenizer.pad_token_id or 0)
            loss = alpha * dl + (1 - alpha) * hl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); el += loss.item(); es += 1
        print(f"  Epoch {epoch+1}/{epochs}  loss={el/max(es,1):.4f}")
    out = MODELS_DIR / "aetheria_distilled.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": student.state_dict(), "vocab_size": v,
                "teacher": KNOWN_MODELS.get(model_key, model_key)}, str(out))
    print(f"\nDistilled -> {out}")


# =============================================================================
# MODE 4: BLIND DEVOUR  (self-play chain — no seed prompts needed)
# =============================================================================

# Short conversation starters to kick off each self-play chain
_BLIND_SEEDS = [
    "Hey.", "What's up?", "yo", "Hello!", "How's it going?",
    "Hey, you there?", "What are you thinking?", "Bored.", "Talk to me.",
    "I'm here.", "What's new?", "Hey, it's me.", "Sup.", "Hi there!",
    "Tell me something.", "I need someone to talk to.", "What do you want to do?",
    "Random question:", "I had a weird day.", "Guess what happened.",
]


def _is_garbled(text: str) -> bool:
    """Return True if the text looks like degraded/gibberish output.
    Triggers on: very long individual words, low alpha ratio, or excessive caps."""
    if not text:
        return True
    words = text.split()
    # single words over 20 chars are almost always garbled (e.g. "nowwwwlllolldontseeooo")
    if any(len(w) > 20 for w in words):
        return True
    # most words over 15 chars = drifting
    long_word_ratio = sum(1 for w in words if len(w) > 15) / max(len(words), 1)
    if long_word_ratio > 0.3:
        return True
    # alpha ratio: real text is mostly letters/spaces
    alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
    if alpha_chars / max(len(text), 1) < 0.65:
        return True
    return False


def gluttony_blind_devour(model_key="dialogpt-small", ckpt_path=None, out_path=None,
                          conversations=25, turns_per_conv=6,
                          epochs=3, lr=3e-4, seq_len=64, batch_size=4,
                          pride_weight=0.4, pride_interval=5, device_str=None):
    """
    Blind self-play devour — teacher talks to itself in chains.
    No fixed seed prompts.  Each AI response feeds back as the next human
    turn, producing hundreds of natural flowing conversation pairs.

    Default model: dialogpt-small (117MB, ~20-30 min on CPU).
    Use --model dialogpt-medium for higher quality (~60-90 min on CPU).
    """
    try:
        from prototype_model import TinyTransformerLM
        import sentencepiece as _spm_module
    except ImportError as e:
        print(f"[Gluttony] Missing: {e}  ->  pip install sentencepiece transformers")
        return

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    total_pairs = conversations * (turns_per_conv - 1)
    print(f"\n{'='*62}")
    print(f"  GLUTTONY — BLIND DEVOUR MODE")
    print(f"  Teacher      : {KNOWN_MODELS.get(model_key, model_key)}")
    print(f"  Device       : {device}")
    print(f"  Conversations: {conversations}  x  {turns_per_conv} turns  = ~{total_pairs} pairs")
    print(f"  Pride anchor : weight={pride_weight}  every {pride_interval} steps")
    print(f"{'='*62}\n")

    if ckpt_path is None:
        ckpt_path = MODELS_DIR / "aetheria_soul.pt"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[Gluttony] Checkpoint not found: {ckpt_path}")
        return

    spm_path = ROOT / "data" / "spm.model"
    persona_path = ROOT / "data" / "persona_seed.txt"
    if not spm_path.exists():
        print(f"[Gluttony] SPM model not found: {spm_path}")
        return
    if not persona_path.exists():
        print(f"[Gluttony] No persona_seed.txt found.")
        return

    out_path = Path(out_path) if out_path else MODELS_DIR / "aetheria_devoured.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load Aetheria ────────────────────────────────────────────────────────
    print("[Pride] Loading Aetheria (the host)...")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    vocab_size = ckpt.get("vocab_size", 0)
    if vocab_size <= 0:
        for k in ("embedding.weight", "tok_emb.weight", "embed.weight"):
            if k in state:
                vocab_size = state[k].shape[0]
                break
    if vocab_size <= 0:
        print("[Gluttony] Cannot determine vocab_size.")
        return

    model = TinyTransformerLM(vocab_size=vocab_size, d_model=256, nhead=4,
                               num_layers=4, dim_ff=1024).to(device)
    model.load_state_dict(state, strict=False)

    sp = _spm_module.SentencePieceProcessor()
    sp.Load(str(spm_path))

    # ── Load Pride ego anchor ─────────────────────────────────────────────
    print("[Pride] Loading ego anchor (persona_seed.txt)...")
    persona_ids = sp.EncodeAsIds(persona_path.read_text(encoding="utf-8", errors="ignore"))
    while len(persona_ids) < seq_len * batch_size * 4:
        persona_ids = persona_ids * 2
    persona_t = torch.tensor(persona_ids, dtype=torch.long)

    def _persona_batch():
        start = torch.randint(0, max(1, len(persona_t) - seq_len - 1), (batch_size,))
        xs = torch.stack([persona_t[s: s + seq_len] for s in start]).to(device)
        ys = torch.stack([persona_t[s + 1: s + seq_len + 1] for s in start]).to(device)
        return xs, ys

    # ── Phase 1: BLIND ABSORB (self-play chain) ───────────────────────────
    print("\n[Gluttony] Phase 1 — BLIND ABSORB (self-play) ...")
    teacher, t_tok = _load_teacher(model_key, str(device))
    eos_token = t_tok.eos_token or "<|endoftext|>"
    eos_id    = t_tok.eos_token_id

    absorbed = []
    seed_cycle = (_BLIND_SEEDS * ((conversations // len(_BLIND_SEEDS)) + 1))[:conversations]

    for conv_i, seed in enumerate(seed_cycle):
        print(f"  [Conv {conv_i+1}/{conversations}] seed: '{seed}'")
        prev_text = seed
        conv_ok = 0

        for turn in range(turns_per_conv):
            # Key fix: only pass the LAST reply as context, not accumulated history.
            # DialoGPT degrades fast when fed its own garbled output back.
            ctx_ids = t_tok.encode(prev_text + eos_token, return_tensors="pt").to(str(device))
            try:
                with torch.no_grad():
                    out = teacher.generate(
                        ctx_ids,
                        max_new_tokens=60,
                        do_sample=True,
                        top_k=50, top_p=0.9, temperature=0.85,
                        pad_token_id=eos_id,
                        repetition_penalty=1.3,
                    )
                input_len = ctx_ids.shape[-1]
                reply_ids  = out[:, input_len:]
                reply_text = t_tok.decode(reply_ids[0], skip_special_tokens=True).strip()

                # Quality gates: length + garble check
                if len(reply_text.split()) < 3:
                    print(f"    T{turn+1}: [skipped — too short]")
                    break
                if _is_garbled(reply_text):
                    print(f"    T{turn+1}: [skipped — garbled: '{reply_text[:50]}']")
                    break  # stop chain immediately, don't absorb noise

                absorbed.append((prev_text, reply_text))
                print(f"    T{turn+1}: {reply_text[:80]}")
                conv_ok += 1
                prev_text = reply_text  # clean reply becomes next prompt

            except Exception as e:
                print(f"    T{turn+1}: [Error: {e}]")
                break

        print(f"    -> {conv_ok} pairs from this conversation")

    del teacher, t_tok
    if str(device) == "cuda":
        torch.cuda.empty_cache()
    print(f"\n[Gluttony] Blind absorbed {len(absorbed)} pairs total.")
    if not absorbed:
        print("[Gluttony] Nothing absorbed — aborting.")
        return

    # ── Phase 2: DIGEST ──────────────────────────────────────────────────
    print("\n[Gluttony] Phase 2 — DIGEST ...")
    abs_ids = []
    for human, ai in absorbed:
        abs_ids.extend(sp.EncodeAsIds(f"Human: {human}\nAetheria: {ai}"))
        eos_sp = sp.eos_id()
        abs_ids.append(eos_sp if eos_sp > 0 else 1)
    while len(abs_ids) < seq_len * batch_size * 10:
        abs_ids = abs_ids * 2
    abs_t = torch.tensor(abs_ids, dtype=torch.long)

    def _abs_batch():
        start = torch.randint(0, max(1, len(abs_t) - seq_len - 1), (batch_size,))
        xs = torch.stack([abs_t[s: s + seq_len] for s in start]).to(device)
        ys = torch.stack([abs_t[s + 1: s + seq_len + 1] for s in start]).to(device)
        return xs, ys

    # ── Phase 3: PURIFY (train + Pride regularisation) ────────────────────
    print(f"\n[Gluttony] Phase 3 — PURIFY (Pride weight={pride_weight}) ...")
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion  = nn.CrossEntropyLoss(ignore_index=0)
    steps_per_epoch = max(20, len(abs_ids) // (seq_len * batch_size))
    model.train()

    for epoch in range(epochs):
        t_loss = a_loss = p_loss = steps = 0
        for step in range(steps_per_epoch):
            ax, ay = _abs_batch()
            al = model(ax)
            v  = al.shape[-1]
            ay = ay.clamp(0, v - 1)
            absorb_loss = criterion(al.view(-1, v), ay.view(-1))

            if pride_weight > 0 and step % pride_interval == 0:
                px, py = _persona_batch()
                pl = model(px)
                py = py.clamp(0, v - 1)
                pride_loss = criterion(pl.view(-1, v), py.view(-1))
                loss = (1 - pride_weight) * absorb_loss + pride_weight * pride_loss
                p_loss += pride_loss.item()
            else:
                loss = absorb_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            a_loss += absorb_loss.item()
            steps += 1

        p_steps = max(steps // max(pride_interval, 1), 1)
        print(f"  Epoch {epoch+1}/{epochs}  total={t_loss/max(steps,1):.4f}  "
              f"absorb={a_loss/max(steps,1):.4f}  pride={p_loss/p_steps:.4f}")

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    meta.update({"model_state_dict": model.state_dict(), "vocab_size": vocab_size,
                 "devoured_from": KNOWN_MODELS.get(model_key, model_key),
                 "pride_weight": pride_weight, "absorbed_pairs": len(absorbed),
                 "mode": "blind_devour"})
    torch.save(meta, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n[Gluttony] Blind devoured model saved -> {out_path}  ({size_mb:.1f} MB)")
    print(f"  Talk: python Original_sin/aetheria_core.py talk --ckpt {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gluttony — devour other models")
    parser.add_argument("--model", default="dialogpt-medium",
                        help=f"Teacher. Known: {list(KNOWN_MODELS)}")
    parser.add_argument("--mode", default="devour",
                        choices=["devour", "generate", "distill", "blind_devour"])
    parser.add_argument("--ckpt", default=str(MODELS_DIR / "aetheria_soul.pt"))
    parser.add_argument("--out", default="")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pride_weight", type=float, default=0.3,
                        help="0=no ego guard, 1=only persona. Default 0.3")
    parser.add_argument("--pride_interval", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    # blind_devour specific
    parser.add_argument("--conversations", type=int, default=25,
                        help="blind_devour: number of self-play conversations (default 25)")
    parser.add_argument("--turns", type=int, default=6,
                        help="blind_devour: turns per conversation (default 6)")
    args = parser.parse_args()

    if args.mode == "devour":
        gluttony_devour(
            model_key=args.model, ckpt_path=Path(args.ckpt),
            out_path=Path(args.out) if args.out else None,
            rounds=args.rounds, epochs=args.epochs, lr=args.lr,
            seq_len=args.seq_len, batch_size=args.batch_size,
            pride_weight=args.pride_weight, pride_interval=args.pride_interval,
            device_str=args.device,
        )
    elif args.mode == "blind_devour":
        gluttony_blind_devour(
            model_key=args.model, ckpt_path=Path(args.ckpt),
            out_path=Path(args.out) if args.out else None,
            conversations=args.conversations, turns_per_conv=args.turns,
            epochs=args.epochs, lr=args.lr,
            seq_len=args.seq_len, batch_size=args.batch_size,
            pride_weight=args.pride_weight, pride_interval=args.pride_interval,
            device_str=args.device,
        )
    elif args.mode == "generate":
        gluttony_collect(model_key=args.model, rounds=args.rounds, out_path=Path(args.output))
    elif args.mode == "distill":
        gluttony_distill(args.model, args.epochs, args.seq_len,
                         args.batch_size, args.temperature, args.alpha)


if __name__ == "__main__":
    main()
