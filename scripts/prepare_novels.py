"""Prepare novel text files for conversational training.

This script extracts quoted dialogue and lines that look like speaker: utterance
from text files placed in `data/novels/` and writes paragraphs to
`data/cleaned_conversations.txt` (one conversation paragraph per block).

Usage:
  python scripts/prepare_novels.py
"""

import re
from pathlib import Path


DATA_DIR = Path(__file__).parents[1] / 'data'
NOVEL_DIR = DATA_DIR / 'novels'
OUT_PATH = DATA_DIR / 'cleaned_conversations.txt'


def extract_dialogue(text: str):
    dialogs = []
    # 1) extract text in double quotes (both straight and curly)
    quotes = re.findall(r'“([^”]+)”|"([^"]+)"', text)
    for a, b in quotes:
        dialogs.append((a or b).strip())

    # 2) extract lines like "Name: speech"
    for line in text.splitlines():
        if ':' in line:
            left, right = line.split(':', 1)
            if len(left.strip()) < 30 and len(right.strip()) > 1:
                dialogs.append(right.strip())

    # filter short
    dialogs = [d for d in dialogs if len(d) > 20]
    return dialogs


def main():
    if not NOVEL_DIR.exists():
        print('No novel files found in', NOVEL_DIR)
        return

    out_paras = []
    for p in sorted(NOVEL_DIR.glob('*.txt')):
        # read with multiple encoding fallbacks to avoid UnicodeDecodeError
        def _read_with_fallback(path):
            for enc in ('utf-8', 'cp1252', 'latin-1'):
                try:
                    return path.read_text(encoding=enc)
                except Exception:
                    continue
            # final fallback: decode bytes and ignore errors
            try:
                return path.read_bytes().decode('utf-8', errors='ignore')
            except Exception:
                return ''

        raw = _read_with_fallback(p)
        dialogs = extract_dialogue(raw)
        # group dialogs in windows of 3-8 utterances to form short conversations
        window = []
        for d in dialogs:
            window.append(d)
            if len(window) >= 4:
                out_paras.append('\n'.join(window))
                window = []
        if window:
            out_paras.append('\n'.join(window))

    if out_paras:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text('\n\n'.join(out_paras), encoding='utf-8')
        print(f'Wrote {len(out_paras)} conversation paragraphs to', OUT_PATH)
    else:
        print('No dialogue found in novels.')


if __name__ == '__main__':
    main()
