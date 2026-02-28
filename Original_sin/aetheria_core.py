"""Aetheria Core — the coordinator that routes between the seven sins.

aetheria_core orchestrates the Original Sin pipeline:

  envy     → calls free AI APIs, saves to data/envy_conversations.txt  (run locally)
  gluttony → distils from a small pre-trained model                    (heavy — use Colab)
  clean    → merges + normalises all data sources
  learn    → trains Aetheria on latest cleaned data                    (heavy — use Colab)
  talk     → launches Lust for a conversation session
  status   → prints counts and checkpoint info
  full     → envy → clean → talk  (local-only safe pipeline)

Usage as a script:
  python Original_sin/aetheria_core.py envy --provider groq --rounds 20
  python Original_sin/aetheria_core.py clean
  python Original_sin/aetheria_core.py talk
  python Original_sin/aetheria_core.py status
  python Original_sin/aetheria_core.py full

  # Heavy work → run on Colab, then copy the .pt back to models/
  python Original_sin/aetheria_core.py gluttony --gluttony_model dialogpt-small
  python Original_sin/aetheria_core.py learn --epochs 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent   # …/Aetheria/


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list, cwd: Path = ROOT):
    """Run a subprocess inheriting the current stdout/stderr."""
    print(f"\n[Core] Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[Core] ⚠  exit code {result.returncode}")
    return result.returncode


def _python(*args):
    """Invoke Python in the same virtual-env Aetheria is running from."""
    return [sys.executable, *[str(a) for a in args]]


# ── pipeline stages ───────────────────────────────────────────────────────────

def envy(provider: str = "groq", rounds: int = 10, mode: str = "collect",
         ckpt: str = ""):
    """Call free AI APIs and save (prompt, response) pairs — runs locally.
    Use mode='teach' to question Aetheria interactively instead of an API."""
    envy_script = ROOT / "Original_sin" / "envy" / "envy.py"
    if mode == "teach":
        print("=== [ENVY] Teach mode — questioning Aetheria locally ===")
        cmd = _python(envy_script, "--mode", "teach", "--rounds", str(rounds))
        if ckpt:
            cmd += ["--ckpt", ckpt]
    else:
        print("=== [ENVY] Calling free AI APIs ===")
        cmd = _python(envy_script, "--mode", "collect",
                      "--provider", provider, "--rounds", str(rounds))
    _run(cmd)


def gluttony(model: str = "dialogpt-medium", rounds: int = 30, mode: str = "devour",
             ckpt: str = "", pride_weight: float = 0.3, epochs: int = 3,
             conversations: int = 25, turns: int = 6):
    """Devour a teacher model into Aetheria.
    Modes: devour (seed prompts), blind_devour (self-play chain), generate, distill.
    blind_devour defaults to dialogpt-small for speed (~20-30 min on CPU)."""
    print(f"=== [GLUTTONY] {mode.upper()} from {model} ===")
    gluttony_script = ROOT / "Original_sin" / "gluttony" / "gluttony.py"
    cmd = _python(gluttony_script, "--model", model, "--mode", mode,
                  "--rounds", str(rounds), "--epochs", str(epochs))
    if mode in ("devour", "blind_devour"):
        base = ckpt or str(ROOT / "models" / "aetheria_soul.pt")
        cmd += ["--ckpt", base, "--pride_weight", str(pride_weight)]
    if mode == "blind_devour":
        cmd += ["--conversations", str(conversations), "--turns", str(turns)]
    _run(cmd)


def clean():
    """Merge and normalise all raw data sources into cleaned_conversations.txt."""
    print("=== [CLEAN] Merging data sources ===")
    raw_path = ROOT / "data" / "conversations.txt"

    # Collect extra sources
    extra_sources = [
        ROOT / "data" / "envy_conversations.txt",
        ROOT / "data" / "gluttony_conversations.txt",
    ]
    merged_lines = []
    for src in extra_sources:
        if src.exists():
            merged_lines.append(src.read_text(encoding="utf-8", errors="ignore"))
            print(f"  + {src.name}")

    if merged_lines:
        # append to conversations.txt so clean_data can process everything at once
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write("\n".join(merged_lines) + "\n")

    clean_script = ROOT / "scripts" / "clean_data.py"
    _run(_python(clean_script))


def learn(epochs: int = 5, batch_size: int = 16, seq_len: int = 128, vocab_size: int = 8000):
    """Fine-tune / train the Aetheria model on the latest cleaned data."""
    print("=== [LEARN] Training the Aetheria model ===")
    data_path = ROOT / "data" / "cleaned_conversations.txt"
    spm_path = ROOT / "data" / "spm.model"

    if not data_path.exists() or data_path.stat().st_size < 100:
        print("[Core] Not enough data. Run `feed` and `clean` first.")
        return

    # (Re-)train SPM if new data changed significantly
    if spm_path.exists():
        print("[Core] Using existing SentencePiece model.")
    else:
        print("[Core] Training SentencePiece tokenizer first …")
        train_spm = ROOT / "scripts" / "train_spm.py"
        _run(_python(train_spm, "--input", data_path, "--model_prefix", ROOT / "data" / "spm", "--vocab_size", str(vocab_size)))

    train_script = ROOT / "scripts" / "prototype_model.py"
    _run(_python(
        train_script,
        "--data", data_path,
        "--spm", spm_path,
        "--vocab_size", str(vocab_size),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seq_len", str(seq_len),
    ))


def talk(top_p: float = 0.85, temperature: float = 0.75, rep_penalty: float = 1.5, ckpt: str = ""):
    """Launch Lust for an interactive conversation with Aetheria."""
    print("=== [LUST] Launching conversation interface ===")
    lust_script = ROOT / "Original_sin" / "lust" / "lust.py"
    cmd = _python(lust_script, "--top_p", str(top_p), "--temperature", str(temperature), "--rep_penalty", str(rep_penalty))
    if ckpt:
        cmd += ["--ckpt", ckpt]
    _run(cmd)


def status():
    """Print data counts and checkpoint info."""
    print("=== [STATUS] Aetheria system ===")
    data_dir = ROOT / "data"
    for fname in ["conversations.txt", "cleaned_conversations.txt", "envy_conversations.txt",
                  "gluttony_conversations.txt", "lust_history.jsonl"]:
        p = data_dir / fname
        if p.exists():
            lines = p.read_text(encoding="utf-8", errors="ignore").count("\n")
            size_kb = p.stat().st_size // 1024
            print(f"  {fname:<40} {lines:>7} lines   {size_kb:>6} KB")
        else:
            print(f"  {fname:<40} (not found)")

    model_dir = ROOT / "models"
    print()
    for pt in sorted(model_dir.glob("*.pt")):
        size_mb = pt.stat().st_size / (1024 * 1024)
        print(f"  checkpoint: {pt.name}  ({size_mb:.1f} MB)")


def retrain(ckpt: str = "", out: str = "", epochs: int = 5, lr: float = 5e-5):
    """Fine-tune a checkpoint on approved/taught feedback pairs."""
    print("=== [RETRAIN] Fine-tuning on feedback.jsonl ===")
    retrain_script = ROOT / "scripts" / "retrain_feedback.py"
    base_ckpt = ckpt or str(ROOT / "models" / "aetheria_soul.pt")
    output    = out  or base_ckpt          # default: overwrite base
    cmd = _python(retrain_script,
                  "--ckpt", base_ckpt,
                  "--out", output,
                  "--epochs", str(epochs),
                  "--lr", str(lr))
    _run(cmd)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aetheria Core — sin coordinator")
    parser.add_argument("command",
                        choices=["envy", "gluttony", "clean", "learn", "talk", "status", "full", "retrain"],
                        help="Command to execute")
    # envy options
    parser.add_argument("--provider", default="groq", help="Envy API provider (groq|huggingface|gemini)")
    parser.add_argument("--envy_mode", default="collect", choices=["collect", "teach"],
                        help="collect=call external API  teach=question Aetheria locally")
    parser.add_argument("--rounds", type=int, default=None, help="Shorthand rounds for envy or gluttony")
    parser.add_argument("--envy_rounds", type=int, default=10)
    # gluttony options
    parser.add_argument("--gluttony_model", default="dialogpt-medium")
    parser.add_argument("--gluttony_rounds", type=int, default=30)
    parser.add_argument("--gluttony_mode", default="devour",
                        choices=["devour", "generate", "distill", "blind_devour"],
                        help="devour=seed prompts | blind_devour=self-play chain | generate=text pairs | distill=KL")
    parser.add_argument("--pride_weight", type=float, default=0.3,
                        help="Corruption guard strength during devour (0=none, 1=only persona)")
    parser.add_argument("--conversations", type=int, default=25,
                        help="blind_devour: number of self-play conversations (default 25)")
    parser.add_argument("--turns", type=int, default=6,
                        help="blind_devour: turns per conversation (default 6)")
    # learn options (heavy — prefer Colab)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=8000)
    # talk / retrain options
    parser.add_argument("--ckpt", default="", help="Path to model checkpoint (for talk / retrain command)")
    parser.add_argument("--out", default="", help="Output checkpoint path (retrain; default: overwrite --ckpt)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for retrain")
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--rep_penalty", type=float, default=1.5)
    args = parser.parse_args()

    if args.command == "envy":
        envy(args.provider,
             args.rounds if args.rounds is not None else args.envy_rounds,
             mode=args.envy_mode,
             ckpt=args.ckpt)
    elif args.command == "gluttony":
        gluttony(args.gluttony_model,
                 args.rounds if args.rounds is not None else args.gluttony_rounds,
                 mode=args.gluttony_mode, ckpt=args.ckpt,
                 pride_weight=args.pride_weight, epochs=args.epochs,
                 conversations=args.conversations, turns=args.turns)
    elif args.command == "clean":
        clean()
    elif args.command == "learn":
        learn(args.epochs, args.batch_size, args.seq_len, args.vocab_size)
    elif args.command == "talk":
        talk(args.top_p, args.temperature, args.rep_penalty, args.ckpt)
    elif args.command == "retrain":
        retrain(args.ckpt, args.out, args.epochs, args.lr)
    elif args.command == "status":
        status()
    elif args.command == "full":
        # local-safe pipeline: Envy → clean → talk
        # (Gluttony + learn are heavy; run those on Colab)
        envy(args.provider, args.envy_rounds)
        clean()
        talk(args.top_p, args.temperature, args.rep_penalty)


if __name__ == "__main__":
    main()
