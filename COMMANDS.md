# Aetheria — Command Reference

All commands are run from the `Aetheria/` folder with the `.venv` active.

```powershell
cd Aetheria
.venv\Scripts\Activate.ps1
```

---

## Quick-start workflows

### Just talk to Aetheria
```powershell
python Original_sin/aetheria_core.py talk
```

### Full local pipeline (collect data → clean → talk)
```powershell
python Original_sin/aetheria_core.py full
```

### Score all models and merge the best
```powershell
python scripts/merge_all.py --dry_run    # preview scores first
python scripts/merge_all.py --top 5      # merge top 5 → models/aetheria_U.pt
```

### Bake your feedback corrections into a new checkpoint
```powershell
python Original_sin/aetheria_core.py retrain
```

---

## `aetheria_core.py` — main entry point

```
python Original_sin/aetheria_core.py <command> [options]
```

### Commands

| Command | What it does | Cost |
|---|---|---|
| `talk` | Chat with Aetheria in the terminal | Free / local |
| `envy` | Calls free AI APIs, saves conversation pairs to `data/envy_conversations.txt` | Free / local |
| `clean` | Merges all data sources and normalises into `data/cleaned_conversations.txt` | Free / local |
| `status` | Prints file sizes/line counts and lists all model checkpoints | Free / local |
| `full` | Runs `envy → clean → talk` in one go | Free / local |
| `retrain` | Fine-tunes a checkpoint on your saved `data/feedback.jsonl` | Free / local |
| `gluttony` | Distils data from a local pre-trained model (e.g. DialoGPT) | **Heavy — use Colab** |
| `learn` | Trains the Aetheria model on `cleaned_conversations.txt` | **Heavy — use Colab** |

---

### `talk` — chat with Aetheria

```powershell
python Original_sin/aetheria_core.py talk
python Original_sin/aetheria_core.py talk --ckpt models/aetheria_soul.pt
python Original_sin/aetheria_core.py talk --temperature 0.8 --top_p 0.9
```

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | `""` (auto-picks latest) | Which `.pt` checkpoint to load |
| `--temperature` | `0.75` | Creativity. Higher = more random, lower = more repetitive |
| `--top_p` | `0.85` | Nucleus sampling cutoff |
| `--rep_penalty` | `1.5` | Penalises repeating the same words |

During the conversation:
- Type your message and press Enter
- Rate each reply `y` / `n` / `s` (skip)
- If you rate `n`, Aetheria retries up to 5 times
- After 5 bad tries you can type a better reply — it gets saved to `corrections.json` and `feedback.jsonl`
- Type `quit` or `exit` to end the session

---

### `envy` — collect training data via free APIs

```powershell
python Original_sin/aetheria_core.py envy
python Original_sin/aetheria_core.py envy --provider groq --rounds 30
```

| Flag | Default | Meaning |
|---|---|---|
| `--provider` | `groq` | API to call: `groq`, `huggingface`, or `gemini` |
| `--rounds` | `10` | How many conversation pairs to collect |
| `--envy_rounds` | `10` | Same as `--rounds` (longer form) |

Requires `GROQ_API_KEY` (or matching key) in environment.

---

### `clean` — normalise data

```powershell
python Original_sin/aetheria_core.py clean
```

No options. Reads `data/conversations.txt`, `data/envy_conversations.txt`, `data/gluttony_conversations.txt` and writes `data/cleaned_conversations.txt`.

---

### `learn` — train the model *(Colab recommended)*

```powershell
python Original_sin/aetheria_core.py learn
python Original_sin/aetheria_core.py learn --epochs 20 --batch_size 32
```

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `5` | Training passes over the data |
| `--batch_size` | `16` | Sequences per gradient step (reduce if OOM) |
| `--seq_len` | `128` | Token context window |
| `--vocab_size` | `8000` | SPM vocabulary size |

Saves to `models/aetheria_ckpt_stepfinal.pt`.

---

### `retrain` — bake feedback into a checkpoint

```powershell
python Original_sin/aetheria_core.py retrain
python Original_sin/aetheria_core.py retrain --ckpt models/aetheria_soul.pt
python Original_sin/aetheria_core.py retrain --ckpt models/aetheria_soul.pt --out models/aetheria_learned.pt --epochs 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | `models/aetheria_soul.pt` | Base model to fine-tune |
| `--out` | same as `--ckpt` | Where to save the result (omit to overwrite in place) |
| `--epochs` | `5` | Fine-tuning passes (more = stronger memory, but risks forgetting) |
| `--lr` | `5e-5` | Learning rate — keep low to avoid overwriting existing knowledge |

Reads from `data/feedback.jsonl` (pairs saved during `talk` sessions).

---

### `gluttony` — devour a teacher model *(recommended: CPU or Colab)*

Gluttony absorbs a pre-trained model's conversational knowledge directly into Aetheria's weights.
Pride's persona seed acts as a corruption guard during absorption — Aetheria stays Aetheria.

```powershell
# Devour DialoGPT-medium into aetheria_soul.pt (recommended first run)
python Original_sin/aetheria_core.py gluttony --ckpt models/aetheria_soul.pt

# Adjust corruption guard (higher = stronger persona preservation)
python Original_sin/aetheria_core.py gluttony --pride_weight 0.5

# Text pairs only (no immediate training — Envy-adjacent)
python Original_sin/aetheria_core.py gluttony --gluttony_mode generate --rounds 100

# Advanced: KL distillation from scratch (builds new model in teacher vocab)
python Original_sin/aetheria_core.py gluttony --gluttony_mode distill
```

| Flag | Default | Meaning |
|---|---|---|
| `--gluttony_model` | `dialogpt-medium` | Teacher to devour: `dialogpt-small/medium`, `gpt2`, `tinyllama`, `qwen` |
| `--gluttony_mode` | `devour` | `devour` = absorb+train \| `generate` = text pairs only \| `distill` = KL from scratch |
| `--gluttony_rounds` | `30` | Seed prompts to ask teacher |
| `--ckpt` | `models/aetheria_soul.pt` | Aetheria checkpoint to absorb INTO |
| `--pride_weight` | `0.3` | Corruption guard: 0 = no guard, 1 = only persona. 0.3 is balanced |
| `--epochs` | `3` | Training epochs after absorption |

Output: `models/aetheria_devoured.pt` — talk to it with `--ckpt models/aetheria_devoured.pt`

**How it works (Rimuru's Predator):**
1. **ABSORB** — asks every seed prompt to DialoGPT, collects its responses
2. **DIGEST** — fine-tunes Aetheria on the absorbed pairs using her own SPM tokenizer
3. **PURIFY** — every 5 steps, Pride runs a batch from `persona_seed.txt` to prevent corruption

---

### `status` — inspect the system

```powershell
python Original_sin/aetheria_core.py status
```

Shows line counts + file sizes for all data files, and lists all `.pt` checkpoints with sizes.

---

## `scripts/merge_all.py` — score and merge all models

Evaluates every `.pt` in `models/` by **perplexity** on persona seed data (lower = better), then weighted-averages the best ones into a single checkpoint.

```powershell
python scripts/merge_all.py --dry_run               # print scores only
python scripts/merge_all.py                          # merge all → models/aetheria_U.pt
python scripts/merge_all.py --top 5                  # only use the 5 best
python scripts/merge_all.py --max_perplexity 300     # skip very bad models
python scripts/merge_all.py --out models/my_best.pt  # custom output path
```

| Flag | Default | Meaning |
|---|---|---|
| `--model_dir` | `models/` | Folder to scan for `.pt` files |
| `--eval_data` | `data/persona_seed_clean.txt` | Text file used to measure perplexity |
| `--spm` | `data/spm.model` | SentencePiece tokenizer |
| `--out` | `models/aetheria_U.pt` | Output merged checkpoint |
| `--max_perplexity` | ∞ | Exclude models with ppl above this number |
| `--top` | `0` (all) | Keep only the N best models |
| `--dry_run` | off | Print rankings without merging |

**What is ppl (perplexity)?**
A score of how well a model predicts Aetheria's voice. Lower is better. Good models sit around 20–100; untrained or mismatched models can reach 500+. Use `--max_perplexity 300` to exclude obvious junk.

---

## `scripts/retrain_feedback.py` — direct fine-tune script

Same as `aetheria_core.py retrain` but with extra control.

```powershell
python scripts/retrain_feedback.py --ckpt models/aetheria_soul.pt
python scripts/retrain_feedback.py --ckpt models/aetheria_soul.pt --out models/aetheria_v2.pt --epochs 10 --lr 1e-4
```

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | *(required)* | Base checkpoint to fine-tune |
| `--out` | same as `--ckpt` | Output path |
| `--feedback` | `data/feedback.jsonl` | Feedback file to train on |
| `--spm` | `data/spm.model` | Tokenizer |
| `--epochs` | `5` | Fine-tuning epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `4` | Batch size |
| `--seq_len` | `64` | Sequence length |
| `--device` | auto | Force `cpu` or `cuda` |

---

## `scripts/prototype_model.py` — low-level training

Used internally by `learn` and `retrain`, but can be called directly.

```powershell
python scripts/prototype_model.py
python scripts/prototype_model.py --data data/persona_seed.txt --epochs 20 --ckpt_name aetheria_soul.pt
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/cleaned_conversations.txt` | Training text file |
| `--spm` | `data/spm.model` | SentencePiece model |
| `--vocab_size` | `4000` | Vocabulary size |
| `--seq_len` | `128` | Context length |
| `--batch_size` | `8` | Batch size |
| `--epochs` | `3` | Training epochs |
| `--device` | auto | `cpu` or `cuda` |
| `--ckpt_name` | `""` | Custom output filename (e.g. `aetheria_soul.pt`) |

---

## `scripts/build_persona_seed.py` — generate persona seed data

Creates `data/persona_seed.txt` — 50 hand-crafted Aetheria dialogues used for perplexity scoring and quick training.

```powershell
python scripts/build_persona_seed.py
```

No flags. Re-run any time to regenerate the file.

---

## Data files explained

| File | Created by | Used by |
|---|---|---|
| `data/conversations.txt` | `prepare_novels.py` | `clean` command |
| `data/envy_conversations.txt` | `envy` command | `clean` command |
| `data/gluttony_conversations.txt` | `gluttony` command | `clean` command |
| `data/cleaned_conversations.txt` | `clean` command | `learn`, `talk` |
| `data/persona_seed.txt` | `build_persona_seed.py` | `merge_all.py` eval, quick training |
| `data/feedback.jsonl` | `talk` sessions | `retrain` command |
| `data/corrections.json` | `talk` sessions | `talk` (instant lookup, checked before model) |
| `data/spm.model` | `train_spm.py` or `learn` | All training + generation scripts |

---

## Recommended workflows

### First-time setup (CPU / local)
```powershell
# 1. Generate persona seed
python scripts/build_persona_seed.py

# 2. Train a small model on it (fast — 2–5 min)
python scripts/prototype_model.py --data data/persona_seed.txt --epochs 20 --ckpt_name aetheria_soul.pt

# 3. Talk to it
python Original_sin/aetheria_core.py talk --ckpt models/aetheria_soul.pt
```

### After collecting more feedback
```powershell
# Bake your taught/approved replies into the model
python Original_sin/aetheria_core.py retrain --ckpt models/aetheria_soul.pt

# Or merge all models and talk to the combined best
python scripts/merge_all.py --top 5
python Original_sin/aetheria_core.py talk --ckpt models/aetheria_U.pt
```

### Heavy training (on Colab)
```
# Use aetheria_o1.ipynb — runs the full O1 pipeline on GPU
# Then download envy.pt / gluttony.pt / aetheria_o1.pt back to models/
```
