"""Dataset preparation for CyberGuard.

- Cleans empty/duplicate records
- Uses deterministic 80/10/10 train/validation/test split
- Converts records into character-level next-token-prediction examples
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharTokenizer, format_record

SEED = 42
CONTEXT_LENGTH = 128


def clean_records(corpus_path: str | Path) -> list[str]:
    seen = set()
    cleaned = []
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = " ".join(str(obj.get("text", "")).replace("\u00a0", " ").split())
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
    return cleaned


def split_records(records: list[str], seed: int = SEED):
    items = records[:]
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


class CybersecurityQADataset(Dataset):
    def __init__(self, records, tokenizer: CharTokenizer, context_length: int = CONTEXT_LENGTH):
        self.records = records
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.samples = []
        for text in records:
            ids = tokenizer.encode(format_record(text))
            # Need context_length + 1 chars for shifted input/target.
            if len(ids) < 2:
                continue
            # Chunk long records; pad short records.
            start = 0
            while start < len(ids) - 1:
                chunk = ids[start:start + context_length + 1]
                if len(chunk) < context_length + 1:
                    chunk = chunk + [tokenizer.pad_id] * (context_length + 1 - len(chunk))
                self.samples.append(chunk)
                if start + context_length + 1 >= len(ids):
                    break
                start += context_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = torch.tensor(self.samples[idx], dtype=torch.long)
        return seq[:-1], seq[1:]


def build_dataloaders(corpus_path, vocab_path, batch_size=16, context_length=128):
    records = clean_records(corpus_path)
    train_r, val_r, test_r = split_records(records)
    tokenizer = CharTokenizer.load(vocab_path)
    train_ds = CybersecurityQADataset(train_r, tokenizer, context_length)
    val_ds = CybersecurityQADataset(val_r, tokenizer, context_length)
    test_ds = CybersecurityQADataset(test_r, tokenizer, context_length)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return tokenizer, (train_r, val_r, test_r), (train_loader, val_loader, test_loader)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    corpus = root / "data" / "corpus.jsonl"
    vocab = root / "data" / "vocab.json"
    if not vocab.exists():
        CharTokenizer.from_corpus(corpus).save(vocab)
    tokenizer, splits, loaders = build_dataloaders(corpus, vocab)
    tr, va, te = splits
    print("DATASET TEST PASSED")
    print(f"Clean records: {len(tr)+len(va)+len(te)}")
    print(f"Train/Validation/Test records: {len(tr)}/{len(va)}/{len(te)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    xb, yb = next(iter(loaders[0]))
    print("Batch input shape:", tuple(xb.shape))
    print("Batch target shape:", tuple(yb.shape))
