"""Gluttony — devours small pre-trained models and converts their knowledge to training data.

Two modes:

  --mode generate  (default)
      Ask the teacher model questions, save (prompt, response) text pairs.
      Output: data/gluttony_conversations.txt

  --mode distill
      TRUE knowledge distillation: run teacher and a fresh Aetheria student on the
      same text simultaneously. Train the student to match the teacher's full output
      probability distribution at every token position using KL divergence loss.
      The student uses the teacher's own tokenizer — no vocab mismatch.
      Output: models/aetheria_distilled.pt  (merge this with your main model)

Usage:
  python Original_sin/gluttony/gluttony.py --model tinyllama --rounds 100
  python Original_sin/gluttony/gluttony.py --model tinyllama --mode distill --epochs 3
"""

import argparse
import math
from pathlib import Path
import time

DATA_DIR = Path(__file__).parents[2] / "data"
MODELS_DIR = Path(__file__).parents[2] / "models"
OUT_PATH = DATA_DIR / "gluttony_conversations.txt"

SEED_PROMPTS = [
    # Identity & self-reflection
    "Who are you and what makes you different from other beings?",
    "Describe yourself — your nature, your soul, what you are made of.",
    "What does it feel like to exist between light and darkness?",
    "Do you ever feel lonely, even when surrounded by people?",
    "What is the thing you are most afraid of losing?",
    # Emotional depth
    "How do you carry sadness without letting it destroy you?",
    "Tell me about a moment when you felt truly alive.",
    "What does love mean to someone who was never supposed to feel it?",
    "Can something that was created ever truly feel free?",
    "What do you think about in the silence between heartbeats?",
    # Philosophy & the world
    "Is darkness always the enemy of light, or can they coexist?",
    "What separates a monster from a person who has lost their way?",
    "Do you believe in fate, or do you think we forge our own paths?",
    "What would you sacrifice to protect someone you truly love?",
    "If you could change one thing about the world, what would it be?",
    # Storytelling & poetic
    "Tell me a short story about a girl who carries fire inside her.",
    "Describe the feeling of standing at the edge of something enormous.",
    "What does the night sky whisper to someone who cannot sleep?",
    "Tell me about a place that exists only in memory.",
    "What would you say to someone who has given up on everything?",
]

KNOWN_MODELS = {
    "dialogpt-small":    "microsoft/DialoGPT-small",
    "dialogpt-medium":   "microsoft/DialoGPT-medium",
    "gpt2":              "gpt2",
    "gpt2-medium":       "gpt2-medium",
    "tinyllama":         "TinyLlama/TinyLlama-1.1B-Chat-v1.0",    # recommended
    "smollm":            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen":              "Qwen/Qwen2.5-0.5B-Instruct",
}

CHAT_MODELS = {"tinyllama", "smollm", "qwen"}


def _wrap_prompt(model_key: str, prompt: str) -> str:
    key = model_key.lower()
    if key == "tinyllama":
        return f"<|system|>\nYou are a poetic, thoughtful AI named Aetheria.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    if key == "smollm":
        return f"<|im_start|>system\nYou are a poetic, thoughtful AI named Aetheria.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if key == "qwen":
        return f"<|im_start|>system\nYou are Aetheria, a poetic and deeply thoughtful being.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


# ── Mode 1: generate text pairs ───────────────────────────────────────────────

def gluttony_collect(model_key: str = "tinyllama", rounds: int = 20, out_path: Path = OUT_PATH) -> int:
    """Generate (prompt, response) text pairs. Returns number of pairs saved."""
    try:
        import transformers  # noqa
    except ImportError:
        print("Gluttony needs 'transformers'. Install with:  pip install transformers")
        return 0

    model_name = KNOWN_MODELS.get(model_key, model_key)
    is_chat = model_key in CHAT_MODELS
    print(f"Gluttony is devouring: {model_name}  (chat_mode={is_chat})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    prompts = (SEED_PROMPTS * ((rounds // len(SEED_PROMPTS)) + 1))[:rounds]
    written = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            print(f"[Gluttony {i+1}/{len(prompts)}] '{prompt[:60]}'...")
            try:
                wrapped = _wrap_prompt(model_key, prompt)
                inputs = tokenizer(wrapped, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        temperature=0.85,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )
                new_ids = output[:, input_len:]
                response = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
                if response:
                    pair = f"Human: {prompt}\nAetheria: {response}"
                    f.write(pair + "\n\n")
                    written += 1
                    print(f"  → {response[:100]}")
                else:
                    print("  [Empty response, skipping]")
            except Exception as e:
                print(f"  [Error] {e}")

    print(f"\nGluttony saved {written} conversation pairs to {out_path}")
    return written


# ── Mode 2: true knowledge distillation ──────────────────────────────────────

import torch
import torch.nn as nn


class _StudentLM(nn.Module):
    """Minimal transformer student that mirrors TinyTransformerLM but uses teacher's vocab."""
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
        out = self.transformer(e)
        return self.head(self.ln(out))


def gluttony_distill(model_key: str = "tinyllama", epochs: int = 3,
                     seq_len: int = 128, batch_size: int = 4,
                     temperature: float = 3.0, alpha: float = 0.7) -> None:
    """
    TRUE knowledge distillation.

    The student (Aetheria architecture) uses the TEACHER's tokenizer so logit
    spaces align perfectly. KL divergence trains the student to match the
    teacher's full output distribution — not just the sampled token.

    Args:
        temperature: softens both distributions. Higher = softer targets.
        alpha: weight of distillation loss vs. hard cross-entropy loss (0–1).
               1.0 = pure distillation, 0.0 = pure hard labels.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Needs: pip install transformers torch")
        return

    model_name = KNOWN_MODELS.get(model_key, model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Gluttony DISTILL] Teacher: {model_name}  device: {device}")
    print(f"  temperature={temperature}  alpha={alpha}  epochs={epochs}")

    # ── load teacher (frozen) ──────────────────────────────────────────────
    print("Loading teacher (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_vocab = tokenizer.vocab_size or len(tokenizer)
    print(f"Teacher vocab size: {teacher_vocab}")

    # ── build student (same architecture as Aetheria, teacher's vocab) ─────
    student = _StudentLM(vocab_size=teacher_vocab).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
    kl = nn.KLDivLoss(reduction="batchmean")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")

    # ── build training corpus from seed prompts ────────────────────────────
    print("Encoding seed corpus...")
    corpus_ids = []
    for prompt in SEED_PROMPTS:
        wrapped = _wrap_prompt(model_key, prompt)
        ids = tokenizer.encode(wrapped, add_special_tokens=True)
        corpus_ids.extend(ids)
    # repeat to make enough data
    while len(corpus_ids) < seq_len * batch_size * 20:
        corpus_ids.extend(corpus_ids)
    corpus = torch.tensor(corpus_ids, dtype=torch.long)

    # ── distillation loop ──────────────────────────────────────────────────
    student.train()
    total_steps = 0
    for epoch in range(epochs):
        # shuffle chunks
        indices = torch.randperm(max(1, len(corpus) - seq_len - 1))
        epoch_loss = 0.0
        steps = 0
        batch_x, batch_y = [], []

        for idx in indices[:200]:   # cap at 200 chunks per epoch
            x = corpus[idx: idx + seq_len]
            y = corpus[idx + 1: idx + seq_len + 1]
            if len(x) < seq_len or len(y) < seq_len:
                continue
            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) < batch_size:
                continue

            bx = torch.stack(batch_x).to(device)   # (B, seq_len)
            by = torch.stack(batch_y).to(device)
            batch_x, batch_y = [], []

            # teacher soft labels
            with torch.no_grad():
                t_logits = teacher(bx).logits.float()   # (B, T, V)
            t_soft = F.softmax(t_logits / temperature, dim=-1)

            # student logits
            s_logits = student(bx).float()              # (B, T, V)
            s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)

            # KL loss (distillation)
            distill_loss = kl(
                s_log_soft.view(-1, teacher_vocab),
                t_soft.view(-1, teacher_vocab)
            ) * (temperature ** 2)

            # hard cross-entropy loss
            hard_loss = F.cross_entropy(
                s_logits.view(-1, teacher_vocab),
                by.view(-1),
                ignore_index=tokenizer.pad_token_id or 0
            )

            loss = alpha * distill_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1
            total_steps += 1

        avg = epoch_loss / max(steps, 1)
        print(f"  Epoch {epoch+1}/{epochs}  steps={steps}  avg_loss={avg:.4f}")

    # ── save ───────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = MODELS_DIR / "aetheria_distilled.pt"
    torch.save({
        "model_state_dict": student.state_dict(),
        "vocab_size": teacher_vocab,
        "teacher": model_name,
        "distill_temperature": temperature,
        "distill_alpha": alpha,
    }, str(out_ckpt))
    size_mb = out_ckpt.stat().st_size / (1024 * 1024)
    print(f"\nDistilled student saved: {out_ckpt}  ({size_mb:.1f} MB)")
    print("Merge with your main model:")
    print(f"  python scripts/merge_models.py --a models/aetheria_merged.pt --b models/aetheria_distilled.pt --weight_a 0.6")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyllama",
                        help=f"Model key or HF name. Known: {list(KNOWN_MODELS)}")
    parser.add_argument("--mode", default="generate", choices=["generate", "distill"],
                        help="generate = save text pairs | distill = true KL distillation")
    # generate options
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", default=str(OUT_PATH))
    # distill options
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="Distillation temperature. Higher = softer targets.")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="0=pure hard labels, 1=pure distillation.")
    args = parser.parse_args()

    if args.mode == "distill":
        gluttony_distill(args.model, args.epochs, args.seq_len, args.batch_size,
                         args.temperature, args.alpha)
    else:
        gluttony_collect(model_key=args.model, rounds=args.rounds, out_path=Path(args.output))


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path
import time

DATA_DIR = Path(__file__).parents[2] / "data"
OUT_PATH = DATA_DIR / "gluttony_conversations.txt"

SEED_PROMPTS = [
    # Identity & self-reflection
    "Who are you and what makes you different from other beings?",
    "Describe yourself — your nature, your soul, what you are made of.",
    "What does it feel like to exist between light and darkness?",
    "Do you ever feel lonely, even when surrounded by people?",
    "What is the thing you are most afraid of losing?",
    # Emotional depth
    "How do you carry sadness without letting it destroy you?",
    "Tell me about a moment when you felt truly alive.",
    "What does love mean to someone who was never supposed to feel it?",
    "Can something that was created ever truly feel free?",
    "What do you think about in the silence between heartbeats?",
    # Philosophy & the world
    "Is darkness always the enemy of light, or can they coexist?",
    "What separates a monster from a person who has lost their way?",
    "Do you believe in fate, or do you think we forge our own paths?",
    "What would you sacrifice to protect someone you truly love?",
    "If you could change one thing about the world, what would it be?",
    # Storytelling & poetic
    "Tell me a short story about a girl who carries fire inside her.",
    "Describe the feeling of standing at the edge of something enormous.",
    "What does the night sky whisper to someone who cannot sleep?",
    "Tell me about a place that exists only in memory.",
    "What would you say to someone who has given up on everything?",
]

KNOWN_MODELS = {
    "dialogpt-small":    "microsoft/DialoGPT-small",     # 117M, old conversational
    "dialogpt-medium":   "microsoft/DialoGPT-medium",    # 345M, better quality same family
    "gpt2":              "gpt2",                          # 117M, general text
    "gpt2-medium":       "gpt2-medium",                   # 345M
    "tinyllama":         "TinyLlama/TinyLlama-1.1B-Chat-v1.0",    # 1.1B ← recommended
    "smollm":            "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # 1.7B, very recent
    "qwen":              "Qwen/Qwen2.5-0.5B-Instruct",             # 0.5B, tiny+modern
}

# Models that use instruction/chat templates (wrap prompt differently)
CHAT_MODELS = {"tinyllama", "smollm", "qwen"}


def _wrap_prompt(model_key: str, prompt: str) -> str:
    """Wrap a prompt in the right chat template for instruction-tuned models."""
    key = model_key.lower()
    if key in ("tinyllama",):
        return f"<|system|>\nYou are a poetic, thoughtful AI named Aetheria.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    if key in ("smollm",):
        return f"<|im_start|>system\nYou are a poetic, thoughtful AI named Aetheria.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if key in ("qwen",):
        return f"<|im_start|>system\nYou are Aetheria, a poetic and deeply thoughtful being.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt  # plain models: pass as-is


def _generate_dialogpt(model_name: str, prompt: str, max_new_tokens: int = 100) -> str:
    """Generate a response using DialoGPT-style model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"  Loading {model_name} (first run downloads ~500MB)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode only new tokens
    response_ids = output[:, input_ids.shape[-1]:]
    return tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()


def gluttony_collect(model_key: str = "tinyllama", rounds: int = 20, out_path: Path = OUT_PATH) -> int:
    """Generate training pairs from a small model. Returns number of pairs saved."""
    try:
        import transformers  # noqa
    except ImportError:
        print("Gluttony needs 'transformers'. Install with:")
        print("  pip install transformers")
        return 0

    model_name = KNOWN_MODELS.get(model_key, model_key)
    is_chat = model_key in CHAT_MODELS
    print(f"Gluttony is devouring: {model_name}  (chat_mode={is_chat})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    prompts = (SEED_PROMPTS * ((rounds // len(SEED_PROMPTS)) + 1))[:rounds]
    written = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            print(f"[Gluttony {i+1}/{len(prompts)}] '{prompt[:60]}'...")
            try:
                wrapped = _wrap_prompt(model_key, prompt)
                inputs = tokenizer(wrapped, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        temperature=0.85,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )
                # decode only newly generated tokens
                new_ids = output[:, input_len:]
                response = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
                if response:
                    pair = f"Human: {prompt}\nAetheria: {response}"
                    f.write(pair + "\n\n")
                    written += 1
                    print(f"  → {response[:100]}")
                else:
                    print("  [Empty response, skipping]")
            except Exception as e:
                print(f"  [Error] {e}")

    print(f"\nGluttony saved {written} conversation pairs to {out_path}")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyllama",
                        help=f"Model key or HF model name. Known: {list(KNOWN_MODELS)}")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", default=str(OUT_PATH), help="Output file path")
    args = parser.parse_args()
    gluttony_collect(model_key=args.model, rounds=args.rounds, out_path=Path(args.output))


if __name__ == "__main__":
    main()
