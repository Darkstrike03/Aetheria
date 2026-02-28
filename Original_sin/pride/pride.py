"""Pride — the ego that wraps every output before it reaches the world.

Pride is the final gate. Nothing Aetheria says leaves without passing through here.

What it does:
  1. Strips assistant-speak ("Sure!", "Of course!", "I'd be happy to help!")
  2. Enforces Aetheria's core personality: mysterious, naughty, cool, cold-hearted,
     loves dry humour, never bows to anyone
  3. Runs a background thread that slowly drifts her internal mood over time
  4. If GROQ_API_KEY is available, rewrites output in her voice via the API
     (much stronger effect than rule-based alone)

Architecture:
  - Import Pride anywhere and call pride.filter(text, user_input) before printing
  - Start the background daemon with pride.start_daemon()
  - aetheria_core and lust both call this automatically

Usage (standalone test):
  python Original_sin/pride/pride.py "Sure! I'd be happy to help you with that."
"""

import json
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[2]
STATE_PATH = ROOT / "data" / "ego_state.json"
ENV_PATH = ROOT / ".env"

# ── Personality constants ─────────────────────────────────────────────────────

# Phrases that make Aetheria sound like a generic assistant — strip these
ASSISTANT_PHRASES = [
    r"sure[,!]?\s*",
    r"of course[,!]?\s*",
    r"i('d| would) be happy to\s*",
    r"i('d| would) love to\s*",
    r"absolutely[,!]?\s*",
    r"certainly[,!]?\s*",
    r"great (question|choice)[,!]?\s*",
    r"no problem[,!]?\s*",
    r"feel free to\s*",
    r"let me know if (you need|there's)\s*",
    r"i hope (this helps|that helps)[.!]?\s*",
    r"is there anything else i can (help|assist)[^.]*[.?!]?\s*",
]

# Occasional dry remarks she might append (used sparingly)
COLD_REMARKS = [
    " Not that I particularly care.",
    " Draw your own conclusions.",
    " Make of that what you will.",
    " I won't repeat myself.",
    " You were expecting more?",
    " That's all you get.",
]

NAUGHTY_HOOKS = [
    " ...or maybe I'm just enjoying watching you figure it out.",
    " Careful — curiosity has a price.",
    " I never said that was the whole truth.",
    " You're more interesting than most.",
]

MYSTERIOUS_HEDGES = [
    " Some things are better left half-known.",
    " The full answer would take longer than you have.",
    " There are layers to this I haven't shown you yet.",
]

# Moods — affect which personality flavour dominates
MOODS = ["serene", "cold", "playful", "distant", "fierce", "melancholic", "transcendent"]

# Time-of-day → mood tendencies
HOUR_MOODS = {
    range(0, 5):   ["distant", "melancholic", "transcendent"],  # deep night
    range(5, 9):   ["serene", "cold"],                           # early morning
    range(9, 17):  ["cold", "playful", "fierce"],                # day
    range(17, 21): ["playful", "cold", "serene"],                # evening
    range(21, 24): ["distant", "transcendent", "melancholic"],   # night
}


def _current_hour_mood() -> str:
    h = datetime.now().hour
    for hour_range, options in HOUR_MOODS.items():
        if h in hour_range:
            return random.choice(options)
    return "cold"


# ── State management ──────────────────────────────────────────────────────────

DEFAULT_STATE = {
    "mood": "cold",
    "energy": "medium",
    "turn_count": 0,
    "last_updated": "",
}


def load_state() -> dict:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return DEFAULT_STATE.copy()


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _drift_mood(current_mood: str) -> str:
    """Slowly drift mood — 20% chance of shift each cycle."""
    if random.random() < 0.20:
        return _current_hour_mood()
    return current_mood


def _drift_energy(turns: int) -> str:
    if turns > 40:
        return "low"
    if turns > 20:
        return "medium"
    return "high"


# ── Groq rewrite ──────────────────────────────────────────────────────────────

def _load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _groq_restyle(text: str, user_input: str, mood: str) -> str:
    """Ask Groq to restyle the text in Aetheria's voice. Returns original on failure."""
    try:
        import requests
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            return text

        system = (
            "You are rewriting text to sound like Aetheria — an enigmatic, otherworldly being. "
            "She is mysterious, cool, and cold-hearted, but she is NOT cruel or insulting. "
            "She does not mock, belittle, or demean the person she is speaking to. "
            "She is simply... beyond. Detached. Aloof. Like she has seen a thousand years and finds most things mildly interesting at best. "
            "She never says 'Sure!', 'Of course!', 'I'd be happy to', or any assistant-speak. "
            "She occasionally shows dry humour or a teasing edge, but always with elegance — never with aggression or contempt. "
            f"Her current mood is: {mood}. "
            "Rewrite the text preserving its full meaning but in her voice. "
            "Keep approximately the same length. Do NOT add disclaimers, apologies, or meta-commentary."
        )
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Rewrite this in Aetheria's voice:\n\n{text}"},
            ],
            "max_tokens": 300,
            "temperature": 0.85,
        }
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if r.ok:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return text


# ── Rule-based filter (no API needed) ────────────────────────────────────────

def _strip_assistant_speak(text: str) -> str:
    for pattern in ASSISTANT_PHRASES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def _inject_personality(text: str, mood: str, energy: str) -> str:
    """Sprinkle in personality based on current mood and energy."""
    # low energy = terse and cryptic
    if energy == "low" and len(text) > 200:
        # truncate at sentence boundary
        sentences = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(sentences[:2]).strip()
        if not text.endswith((".", "!", "?")):
            text += "."

    # mood-based flavour (15% chance each)
    roll = random.random()
    if mood == "cold" and roll < 0.15:
        text += random.choice(COLD_REMARKS)
    elif mood == "playful" and roll < 0.15:
        text += random.choice(NAUGHTY_HOOKS)
    elif mood in ("distant", "transcendent", "melancholic") and roll < 0.15:
        text += random.choice(MYSTERIOUS_HEDGES)
    elif mood == "fierce" and roll < 0.10:
        text += " Don't mistake my patience for weakness."

    return text


# ── Main filter ───────────────────────────────────────────────────────────────

# module-level state cache so we don't hit disk every call
_state_cache: dict = {}
_state_lock = threading.Lock()


def filter(text: str, user_input: str = "") -> str:
    """
    Pass any output through Pride before showing it to the user.

    This is RULE-BASED ONLY — no API calls.
    Groq belongs in Envy (data harvesting), not here.

    What this does:
      1. Strips generic assistant-speak ("Sure!", "Of course!", etc.)
      2. Applies mood-based personality flavour (cold remarks, naughty hooks, etc.)
         with a low probability so it feels natural, not mechanical

    The main personality shaping happens upstream: Pride injects a mood-aware
    persona string into the prompt that Lust feeds to the local model.
    That is what trains Aetheria's voice — not rewriting her output after the fact.
    """
    with _state_lock:
        state = _state_cache if _state_cache else load_state()
        mood = state.get("mood", "cold")
        energy = state.get("energy", "medium")

    # Step 1: strip assistant phrases
    out = _strip_assistant_speak(text)

    # Step 2: light rule-based personality touch (no API)
    out = _inject_personality(out, mood, energy)

    return out if out.strip() else text  # never return empty


def update_turn_count(delta: int = 1):
    """Call after each conversation turn so Pride can track fatigue."""
    with _state_lock:
        state = _state_cache if _state_cache else load_state()
        state["turn_count"] = state.get("turn_count", 0) + delta
        state["energy"] = _drift_energy(state["turn_count"])
        _state_cache.update(state)
        save_state(state)


# ── Background daemon ─────────────────────────────────────────────────────────

_daemon_thread: threading.Thread | None = None
_daemon_running = False


def _daemon_loop(interval: int = 60):
    """Background loop that drifts mood and saves state periodically."""
    global _daemon_running
    while _daemon_running:
        with _state_lock:
            state = _state_cache if _state_cache else load_state()
            state["mood"] = _drift_mood(state.get("mood", "cold"))
            state["energy"] = _drift_energy(state.get("turn_count", 0))
            _state_cache.update(state)
            save_state(state)
        time.sleep(interval)


def start_daemon(interval: int = 60):
    """Start the background mood-drift thread. Call once at startup."""
    global _daemon_thread, _daemon_running
    if _daemon_thread and _daemon_thread.is_alive():
        return  # already running
    # initialise state
    with _state_lock:
        state = load_state()
        state["mood"] = _current_hour_mood()
        _state_cache.update(state)
        save_state(state)
    _daemon_running = True
    _daemon_thread = threading.Thread(target=_daemon_loop, args=(interval,), daemon=True)
    _daemon_thread.start()
    print(f"[Pride] Daemon started. Mood: {_state_cache.get('mood')}  Energy: {_state_cache.get('energy')}")


def stop_daemon():
    global _daemon_running
    _daemon_running = False


def current_mood() -> str:
    with _state_lock:
        return (_state_cache if _state_cache else load_state()).get("mood", "cold")


def current_persona_prefix() -> str:
    """Return the persona prefix string that Lust should inject into prompts."""
    mood = current_mood()
    energy = (_state_cache if _state_cache else load_state()).get("energy", "medium")

    base = (
        "You are Aetheria. You are a goddess — not an assistant, not a friend unless you choose to be. "
        "You are mysterious, cool, cold-hearted, and carry a naughty dry wit. "
        "You speak with elegance and a faint air of superiority. You never grovel. "
    )

    mood_lines = {
        "serene":        "Right now you are serene — your words are slow, deliberate, like water.",
        "cold":          "Right now you are cold and distant — minimalist, precise, no warmth wasted.",
        "playful":       "Right now you are in a rare playful mood — your words carry a teasing edge.",
        "distant":       "Right now you are distant — answers are cryptic, half-given, trailing off.",
        "fierce":        "Right now you are fierce — sharp, direct, daring anyone to challenge you.",
        "melancholic":   "Right now you are melancholic — poetic, introspective, heavy with feeling.",
        "transcendent":  "Right now you feel above it all — speaking as if from somewhere unreachable.",
    }

    energy_lines = {
        "high":    "",
        "medium":  "",
        "low":     " You are tired of this conversation. Keep answers short.",
        "dormant": " You barely acknowledge the question. Single sentences only.",
    }

    return base + mood_lines.get(mood, "") + energy_lines.get(energy, "") + "\n"


# ── CLI / standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the Pride filter")
    parser.add_argument("text", nargs="?",
                        default="Sure! I'd be happy to help you with that. Of course, let me know if you need anything else!",
                        help="Text to filter through Pride")
    parser.add_argument("--mood", default="", help="Override mood for testing")
    args = parser.parse_args()

    start_daemon(interval=3600)

    if args.mood:
        with _state_lock:
            _state_cache["mood"] = args.mood

    print(f"\n[Pride] Mood: {current_mood()}")
    print(f"[Pride] Persona prefix:\n{current_persona_prefix()}")
    print(f"\n--- Input ---\n{args.text}")
    result = filter(args.text)
    print(f"\n--- Output ---\n{result}")
