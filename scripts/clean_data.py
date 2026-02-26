"""Clean raw scraped text into paragraph conversations suitable for training.

Produces `data/cleaned_conversations.txt` where each paragraph is separated by a blank line.
"""

import re
from pathlib import Path


DATA_DIR = Path(__file__).parents[1] / 'data'
RAW_PATH = DATA_DIR / 'conversations.txt'
CLEAN_PATH = DATA_DIR / 'cleaned_conversations.txt'


def clean_text(raw: str) -> list:
    # remove control characters
    raw = re.sub(r"[\x00-\x1f\x7f]+", " ", raw)
    # normalize whitespace
    raw = re.sub(r"\s+", " ", raw)
    # keep typical punctuation but remove other non-printables
    raw = raw.strip()
    # split into paragraphs by double newlines or long breaks
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if len(p.strip()) > 20]
    return paras


def main():
    if not RAW_PATH.exists():
        print("No raw scraped text found at", RAW_PATH)
        return
    raw = RAW_PATH.read_text(encoding='utf-8')
    paras = clean_text(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.write_text("\n\n".join(paras), encoding='utf-8')
    print(f"Wrote {len(paras)} cleaned paragraphs to {CLEAN_PATH}")


if __name__ == '__main__':
    main()