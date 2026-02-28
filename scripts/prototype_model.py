"""Prototype Aetheria core: tiny transformer language model trainer.

This script is a minimal, self-contained training loop intended for
experimentation on small datasets and limited hardware (CPU or a small GPU).

Behavior:
- If `data/cleaned_conversations.txt` exists, it will train a SentencePiece
  tokenizer and prepare data; otherwise it exits with an instruction.
- Builds a small Transformer LM from scratch (PyTorch) and trains for a
  configurable small number of steps.

This is a learning / prototyping scaffold — not production code.
"""

import os
import math
import random
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import re
from tqdm import tqdm


DATA_PATH = Path(__file__).parents[1] / "data" / "cleaned_conversations.txt"
SP_MODEL = Path(__file__).parents[1] / "data" / "spm.model"


class SimpleCharTokenizer:
    """Very small fallback tokenizer that maps characters to ids.

    Reserves: 0=pad, 1=unk, 2=<s>, 3=</s>, chars start at 4.
    """
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char2id = {c: i + 4 for i, c in enumerate(chars)}
        self.id2char = {i: c for c, i in self.char2id.items()}

    def encode(self, text: str):
        ids = []
        for ch in text:
            ids.append(self.char2id.get(ch, 1))
        return ids

    def pad_id(self):
        return 0

    def eos_id(self):
        return 3

    def __len__(self):
        return 4 + len(self.char2id)



class TextDataset(Dataset):
    def __init__(self, ids, seq_len=128):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.ids[idx: idx + self.seq_len]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq)
        emb = self.token_emb(x)  # (batch, seq, d)
        emb = self.pos_enc(emb)
        # Transformer expects (seq, batch, d)
        out = self.transformer(emb.transpose(0, 1))
        out = out.transpose(0, 1)
        out = self.ln(out)
        logits = self.head(out)
        return logits


def build_or_load_spm(data_file: Path, model_path: Path, vocab_size=8000):
    if model_path.exists():
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        return sp

    print("Training SentencePiece tokenizer...")
    # attempt to pick a sensible vocab_size based on corpus unique tokens
    try:
        raw = data_file.read_text(encoding='utf-8')
        words = set(re.findall(r"\S+", raw))
        # reserve a few special tokens (<unk>, <s>, </s>)
        max_from_corpus = max(2, len(words) + 3)
        if vocab_size > max_from_corpus:
            print(f"Requested vocab_size={vocab_size} larger than corpus unique tokens; reducing to {max_from_corpus}")
            vocab_size = max_from_corpus
    except Exception:
        # if reading fails, fall back to the provided value and let SPM error if needed
        pass

    train_cmd = f"--input={data_file} --model_prefix={model_path.with_suffix('')} --vocab_size={vocab_size} --model_type=bpe"
    try:
        spm.SentencePieceTrainer.Train(train_cmd)
    except RuntimeError as e:
        msg = str(e)
        m = re.search(r"Please set it to a value <=\s*(\d+)", msg)
        if m:
            allowed = int(m.group(1))
            if allowed <= 0:
                raise
            print(f"Requested vocab_size={vocab_size} too large; retrying with vocab_size={allowed}")
            train_cmd = f"--input={data_file} --model_prefix={model_path.with_suffix('')} --vocab_size={allowed} --model_type=bpe"
            spm.SentencePieceTrainer.Train(train_cmd)
        else:
            raise
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def prepare_data(sp, data_file: Path, seq_len=128):
    text = data_file.read_text(encoding='utf-8')
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    tokens = []
    for line in lines:
        ids = sp.encode(line)
        tokens.extend(ids + [sp.pad_id() if sp.pad_id() != -1 else sp.eos_id()])
    return tokens


def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return x, y


def _save_checkpoint(model, out_dir: Path, sp, vocab_size, step=None, epoch=None, ckpt_name: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_name:
        ckpt_path = out_dir / ckpt_name
    else:
        ckpt_path = out_dir / (f"aetheria_ckpt_step{step}.pt" if step is not None else "aetheria_ckpt.pt")
    tok_meta = {}
    try:
        from shutil import copy2
        if hasattr(sp, 'char2id'):
            tok_path = out_dir / 'tokenizer_char.json'
            import json
            json.dump({'type': 'char', 'char2id': sp.char2id}, open(tok_path, 'w', encoding='utf-8'))
            tok_meta = {'type': 'char', 'path': str(tok_path)}
        else:
            spm_src = Path(__file__).parents[1] / 'data' / 'spm.model'
            spm_vocab = Path(__file__).parents[1] / 'data' / 'spm.vocab'
            if spm_src.exists():
                copy2(spm_src, out_dir / 'spm.model')
                tok_meta = {'type': 'spm', 'path': str(out_dir / 'spm.model')}
                if spm_vocab.exists():
                    copy2(spm_vocab, out_dir / 'spm.vocab')
    except Exception:
        tok_meta = {}

    torch.save({'model_state_dict': model.state_dict(), 'vocab_size': vocab_size, 'tokenizer': tok_meta, 'step': step, 'epoch': epoch}, ckpt_path)
    # also write latest
    copy_latest = out_dir / 'aetheria_latest.pt'
    try:
        copy2(ckpt_path, copy_latest)
    except Exception:
        pass


def train_loop(model, dataloader, optimizer, device, epochs=3, log_every=50, save_every_steps: int = 0, save_dir: Path = None, sp=None, vocab_size=None):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    global_step = 0
    try:
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                if global_step % log_every == 0:
                    pbar.set_postfix(loss=loss.item())
                if save_every_steps and save_every_steps > 0 and global_step % save_every_steps == 0 and save_dir is not None:
                    print(f"Saving checkpoint at step {global_step}...")
                    _save_checkpoint(model, save_dir, sp, vocab_size, step=global_step, epoch=epoch+1)
    except KeyboardInterrupt:
        print('\nTraining interrupted by user; saving current checkpoint...')
        if save_dir is not None:
            _save_checkpoint(model, save_dir, sp, vocab_size, step=global_step, epoch=epoch+1)
        print('Saved interrupt checkpoint. Exiting training loop.')
        return


def main(data_path: Path = DATA_PATH, sp_model: Path = SP_MODEL, vocab_size_arg: int = 4000, seq_len: int = 128, batch_size: int = 8, epochs: int = 3, device_str: str = None, ckpt_name: str = ""):
    if not data_path.exists():
        print("No data found. Please provide `data/cleaned_conversations.txt` (one conversation per paragraph).")
        return

    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Device:', device)

    sp = build_or_load_spm(data_path, sp_model, vocab_size=vocab_size_arg)
    vocab_size = len(sp)

    tokens = prepare_data(sp, data_path, seq_len=seq_len)
    # allow smaller datasets but warn the user; reduce threshold so tiny prototypes can run
    if len(tokens) < 128:
        print('Corpus too small for SPM-based training; switching to char-level fallback.')
        # build simple char tokenizer from raw text and rebuild tokens
        raw = data_path.read_text(encoding='utf-8')
        char_tok = SimpleCharTokenizer(raw)
        # build tokens as concatenation of encoded lines with eos
        tokens = []
        for line in [l.strip() for l in raw.splitlines() if l.strip()]:
            tokens.extend(char_tok.encode(line) + [char_tok.eos_id()])
        vocab_size = len(char_tok)
        sp = char_tok
    else:
        if len(tokens) < 512:
            print('Warning: small dataset (less than 512 tokens). Training may be unstable.')
        vocab_size = len(sp)

    # create sliding windows
    sequences = [tokens[i:i + seq_len] for i in range(0, max(1, len(tokens) - seq_len), seq_len)]
    # pad last sequence if needed
    sequences = [s if len(s) == seq_len else s + [0] * (seq_len - len(s)) for s in sequences]

    dataset = TextDataset([item for seq in sequences for item in seq], seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = TinyTransformerLM(vocab_size=vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # handle resume
    out_dir = Path(__file__).parents[1] / "models"
    out_dir.mkdir(exist_ok=True)
    if hasattr(sp, 'char2id'):
        tok_for_saving = sp
    else:
        tok_for_saving = sp

    train_loop(model, dataloader, optimizer, device, epochs=epochs, log_every=20, save_every_steps=0, save_dir=out_dir, sp=tok_for_saving, vocab_size=vocab_size)

    # final save at end of training
    _save_checkpoint(model, out_dir, tok_for_saving, vocab_size, step='final', epoch=epochs, ckpt_name=ckpt_name)
    out_name = ckpt_name if ckpt_name else 'aetheria_ckpt_stepfinal.pt'
    print('Saved model to', out_dir / out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(DATA_PATH))
    parser.add_argument('--spm', type=str, default=str(SP_MODEL))
    parser.add_argument('--vocab_size', type=int, default=4000)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='Output checkpoint filename inside models/ (e.g. envy.pt). '
                             'Defaults to aetheria_ckpt_stepfinal.pt')
    args = parser.parse_args()
    main(data_path=Path(args.data), sp_model=Path(args.spm), vocab_size_arg=args.vocab_size,
         seq_len=args.seq_len, batch_size=args.batch_size, epochs=args.epochs,
         device_str=args.device, ckpt_name=args.ckpt_name)
