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

def envy(provider: str = "groq", rounds: int = 10):
    """Call free AI APIs and save (prompt, response) pairs — runs locally."""
    print("=== [ENVY] Calling free AI APIs ===")
    envy_script = ROOT / "Original_sin" / "envy" / "envy.py"
    _run(_python(envy_script, "--provider", provider, "--rounds", str(rounds)))


def gluttony(model: str = "dialogpt-small", rounds: int = 30):
    """Distil from a small pre-trained model — heavy, prefer Colab for this."""
    print("=== [GLUTTONY] Distilling from pre-trained model ===")
    gluttony_script = ROOT / "Original_sin" / "gluttony" / "gluttony.py"
    _run(_python(gluttony_script, "--model", model, "--rounds", str(rounds)))


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


def talk(top_p: float = 0.9, temperature: float = 0.85, rep_penalty: float = 1.3, ckpt: str = ""):
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aetheria Core — sin coordinator")
    parser.add_argument("command",
                        choices=["envy", "gluttony", "clean", "learn", "talk", "status", "full"],
                        help="Command to execute")
    # envy options
    parser.add_argument("--provider", default="groq", help="Envy API provider (groq|huggingface|gemini)")
    parser.add_argument("--rounds", type=int, default=None, help="Shorthand rounds for envy or gluttony")
    parser.add_argument("--envy_rounds", type=int, default=10)
    # gluttony options (heavy — prefer Colab)
    parser.add_argument("--gluttony_model", default="dialogpt-small")
    parser.add_argument("--gluttony_rounds", type=int, default=30)
    # learn options (heavy — prefer Colab)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=8000)
    # talk options
    parser.add_argument("--ckpt", default="", help="Path to model checkpoint (for talk command)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--rep_penalty", type=float, default=1.3)
    args = parser.parse_args()

    if args.command == "envy":
        envy(args.provider, args.rounds if args.rounds is not None else args.envy_rounds)
    elif args.command == "gluttony":
        gluttony(args.gluttony_model, args.rounds if args.rounds is not None else args.gluttony_rounds)
    elif args.command == "clean":
        clean()
    elif args.command == "learn":
        learn(args.epochs, args.batch_size, args.seq_len, args.vocab_size)
    elif args.command == "talk":
        talk(args.top_p, args.temperature, args.rep_penalty, args.ckpt)
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
