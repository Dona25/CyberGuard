"""Character-level tokenizer for CyberGuard.

Implements the assignment's required first-version tokenizer using:
    characters = sorted(set(text))
with char_to_id and id_to_char mappings.
"""
from __future__ import annotations
import json
from pathlib import Path

PAD = "<PAD>"
UNK = "<UNK>"


def load_corpus_text(corpus_path: str | Path) -> str:
    """Return the cleaned training text used to build the character vocabulary."""
    corpus_path = Path(corpus_path)
    parts = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = " ".join(str(obj.get("text", "")).split())
            if text:
                parts.append(format_record(text))
    return "\n".join(parts)


def format_record(text: str) -> str:
    """Convert a corpus record into an explicit Q/A language-model format."""
    if "?" in text:
        q, a = text.split("?", 1)
        q = q.strip() + "?"
        a = a.strip()
        return f"Question: {q}\nAnswer: {a}\n"
    return f"Cybersecurity statement: {text.strip()}\n"


class CharTokenizer:
    def __init__(self, characters: list[str]):
        # Special IDs first, followed by assignment-style sorted character set.
        self.characters = sorted(set(characters))
        self.char_to_id = {PAD: 0, UNK: 1}
        for ch in self.characters:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = len(self.char_to_id)
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.pad_id = self.char_to_id[PAD]
        self.unk_id = self.char_to_id[UNK]

    @classmethod
    def from_corpus(cls, corpus_path: str | Path):
        text = load_corpus_text(corpus_path)
        # Required approach: characters = sorted(set(text))
        characters = sorted(set(text))
        return cls(characters)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        out = []
        for idx in ids:
            token = self.id_to_char.get(int(idx), UNK)
            if token not in (PAD, UNK):
                out.append(token)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "character-level",
            "characters": self.characters,
            "char_to_id": self.char_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["characters"])


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    tok = CharTokenizer.from_corpus(root / "data" / "corpus.jsonl")
    tok.save(root / "data" / "vocab.json")
    sample = "Question: What is phishing?\nAnswer:"
    encoded = tok.encode(sample)
    print("Tokenizer type: character-level")
    print("Vocabulary size:", tok.vocab_size)
    print("Sample IDs:", encoded[:30])
    print("Decoded:", tok.decode(encoded))
