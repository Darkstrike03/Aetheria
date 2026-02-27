"""Train a SentencePiece model using the Python API (Windows-friendly).

Usage:
  python scripts/train_spm.py --input data/cleaned_conversations.txt --model_prefix data/spm --vocab_size 8000
"""

import argparse
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model_prefix', required=True)
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--model_type', default='bpe')
    args = parser.parse_args()
    # Inspect corpus and adjust vocab_size to avoid "Vocabulary size too high" errors
    try:
        import re
        raw = open(args.input, 'r', encoding='utf-8').read()
        toks = set(re.findall(r"\S+", raw))
        suggested = max(2, len(toks) + 3)
        if args.vocab_size > suggested:
            print(f"Requested vocab_size={args.vocab_size} larger than unique token estimate={suggested}; reducing to {suggested}")
            args.vocab_size = suggested
    except Exception:
        # if corpus can't be read, fall back to requested value and let trainer error if needed
        pass

    cmd = (f"--input={args.input} --model_prefix={args.model_prefix}"
           f" --vocab_size={args.vocab_size} --model_type={args.model_type}"
           f" --max_sentence_length=524288")
    print('Training SentencePiece with:', cmd)
    try:
        spm.SentencePieceTrainer.Train(cmd)
        print('Done. Model saved as', args.model_prefix + '.model')
    except RuntimeError as e:
        msg = str(e)
        import re
        m = re.search(r"Please set it to a value <=\s*(\d+)", msg)
        if m:
            allowed = int(m.group(1))
            if allowed > 0 and allowed < args.vocab_size:
                print(f"Trainer reported max allowed vocab={allowed}; retrying with that value")
                cmd = (f"--input={args.input} --model_prefix={args.model_prefix}"
                       f" --vocab_size={allowed} --model_type={args.model_type}"
                       f" --max_sentence_length=524288")
                spm.SentencePieceTrainer.Train(cmd)
                print('Done. Model saved as', args.model_prefix + '.model')
                return
        raise


if __name__ == '__main__':
    main()
