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

Usage:
  python Original_sin/envy/envy.py --provider groq --rounds 20
"""

import argparse
import os
import time
import json
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


def envy_collect(provider: str = "groq", rounds: int = 10, pause: float = 1.5,
                 model: str = "") -> int:
    """Collect training pairs from an external AI. Returns number of pairs saved."""
    _load_env()
    call_fn = PROVIDERS.get(provider)
    if call_fn is None:
        print(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    prompts = (SEED_PROMPTS * ((rounds // len(SEED_PROMPTS)) + 1))[:rounds]

    with open(OUT_PATH, "a", encoding="utf-8") as f:
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

    print(f"\nEnvy saved {written} conversation pairs to {OUT_PATH}")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq", choices=list(PROVIDERS))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds between API calls")
    parser.add_argument("--model", default="",
                        help="Override the model name (e.g. llama-3.3-70b-versatile for Groq)")
    args = parser.parse_args()
    envy_collect(provider=args.provider, rounds=args.rounds, pause=args.pause, model=args.model)


if __name__ == "__main__":
    main()
