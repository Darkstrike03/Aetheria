"""Clean raw scraped text into paragraph conversations suitable for training.

Produces `data/cleaned_conversations.txt` where each paragraph is separated by a blank line.
"""

import re
from pathlib import Path


DATA_DIR = Path(__file__).parents[1] / 'data'
RAW_PATH = DATA_DIR / 'conversations.txt'
CLEAN_PATH = DATA_DIR / 'cleaned_conversations.txt'


def clean_text(raw: str) -> list:
    # normalize line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # remove control characters except newlines
    raw = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]+", " ", raw)
    # collapse multiple spaces/tabs within a line (but keep newlines)
    raw = re.sub(r"[^\S\n]+", " ", raw)
    # split into paragraphs on one or more blank lines
    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if len(p.strip()) > 20]
    # also split any single paragraph that's unreasonably long (>2000 chars) at sentence boundaries
    result = []
    for p in paras:
        if len(p) <= 2000:
            result.append(p)
        else:
            # split on sentence endings
            sentences = re.split(r"(?<=[.!?])\s+", p)
            chunk, chunks = "", []
            for s in sentences:
                if len(chunk) + len(s) > 2000 and chunk:
                    chunks.append(chunk.strip())
                    chunk = s
                else:
                    chunk = (chunk + " " + s).strip()
            if chunk:
                chunks.append(chunk)
            result.extend(c for c in chunks if len(c) > 20)
    return result


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