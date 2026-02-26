import argparse
from pathlib import Path
import torch
import torch.nn as nn
import math
import re
import sentencepiece as spm


class SimpleCharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char2id = {c: i + 4 for i, c in enumerate(chars)}
        self.id2char = {i: c for c, i in self.char2id.items()}

    def encode(self, text: str):
        return [self.char2id.get(ch, 1) for ch in text]

    def decode(self, ids):
        return ''.join(self.id2char.get(i, '?') for i in ids)

    def pad_id(self):
        return 0

    def eos_id(self):
        return 3

    def __len__(self):
        return 4 + len(self.char2id)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.token_emb(x)
        emb = self.pos_enc(emb)
        out = self.transformer(emb.transpose(0, 1))
        out = out.transpose(0, 1)
        out = self.ln(out)
        logits = self.head(out)
        return logits


def top_k_sample(logits, k=50, temperature=1.0):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    vals, indices = torch.topk(logits, k)
    probs = torch.nn.functional.softmax(vals / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return int(indices[idx].item())


def build_tokenizer(data_path: Path, spm_path: Path):
    if spm_path.exists():
        sp = spm.SentencePieceProcessor()
        sp.load(str(spm_path))
        return sp
    raw = data_path.read_text(encoding='utf-8')
    return SimpleCharTokenizer(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='models/aetheria_tiny.pt')
    parser.add_argument('--data', type=str, default='data/cleaned_conversations.txt')
    parser.add_argument('--spm', type=str, default='data/spm.model')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    data_path = Path(args.data)
    spm_path = Path(args.spm)

    if not ckpt_path.exists():
        print('Checkpoint not found:', ckpt_path)
        return

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    vocab_size_ckpt = ckpt.get('vocab_size')
    if vocab_size_ckpt is None:
        print('Checkpoint missing vocab_size metadata.')
        return

    tok = build_tokenizer(data_path, spm_path)
    try:
        tok_len = len(tok)
    except Exception:
        # SentencePiece may support len() but fallback
        tok_len = vocab_size_ckpt

    # if tokenizer and checkpoint vocab sizes differ, expand model to fit tokenizer
    target_vocab = max(vocab_size_ckpt, tok_len)
    if target_vocab != vocab_size_ckpt:
        print(f"Warning: checkpoint vocab_size={vocab_size_ckpt} != tokenizer size={tok_len}. Expanding model to {target_vocab} and copying overlapping weights.")

    model = TinyTransformerLM(vocab_size=target_vocab)
    # prepare new state dict by copying overlapping params from checkpoint
    new_state = model.state_dict()
    ckpt_state = ckpt['model_state_dict']
    for k, v in new_state.items():
        if k in ckpt_state:
            v_ckpt = ckpt_state[k]
            if v_ckpt.shape == v.shape:
                new_state[k] = v_ckpt
            else:
                # handle embedding and head mismatch by copying overlapping slices
                if k.endswith('token_emb.weight'):
                    n_rows = min(v_ckpt.shape[0], v.shape[0])
                    new_state[k][:n_rows] = v_ckpt[:n_rows]
                elif k.endswith('head.weight'):
                    # head.weight shape: (out_vocab, d_model)
                    n_rows = min(v_ckpt.shape[0], v.shape[0])
                    new_state[k][:n_rows] = v_ckpt[:n_rows]
                elif k.endswith('head.bias'):
                    n = min(v_ckpt.shape[0], v.shape[0])
                    new_state[k][:n] = v_ckpt[:n]
                else:
                    # leave randomly initialized for mismatched shapes
                    pass
    model.load_state_dict(new_state)
    model.eval()

    print('Interactive mode — type a prompt and press enter. Type \"exit\" to quit.')
    while True:
        try:
            prompt = input('> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.strip().lower() in ('exit', 'quit'):
            break

        if isinstance(tok, SimpleCharTokenizer):
            ids = tok.encode(prompt)
        else:
            ids = tok.encode(prompt)

        ids = ids[:512]
        for _ in range(args.max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            next_logits = logits[0, -1]
            next_id = top_k_sample(next_logits, k=args.top_k, temperature=args.temperature)
            ids.append(next_id)

        if isinstance(tok, SimpleCharTokenizer):
            out = tok.decode(ids)
        else:
            out = tok.decode(ids)
        print(out)


if __name__ == '__main__':
    main()
