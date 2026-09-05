"""CyberGuard response generation.

The Transformer is the primary language model. Because a small from-scratch
character model can produce noisy text before extensive training, CyberGuard
uses a deterministic corpus-answer fallback for questions that closely match
known cybersecurity-awareness records. This improves demo reliability while
keeping the Transformer implementation and generation pipeline intact.
"""
from __future__ import annotations
from pathlib import Path
import json
import re
import torch
from tokenizer import CharTokenizer
from transformer import CyberGuardTransformer


def _load_records(root: Path):
    records = []
    with (root / "data" / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                text = str(obj.get("text", "")).strip()
                if "?" in text:
                    q, a = text.split("?", 1)
                    records.append((q.strip() + "?", a.strip()))
    return records


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "how", "can", "does", "do",
    "why", "when", "where", "which", "to", "for", "of", "in", "on",
    "and", "or", "my", "your", "i", "you", "it", "this", "that",
    "about", "explain", "tell", "me", "please", "related", "relate",
}

ACRONYMS = {
    "mfa": {"mfa", "multi", "factor", "authentication"},
    "2fa": {"2fa", "two", "factor", "authentication"},
    "vpn": {"vpn", "virtual", "private", "network"},
    "ids": {"ids", "intrusion", "detection", "system"},
    "ips": {"ips", "intrusion", "prevention", "system"},
}

def _words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for acronym, expansion in ACRONYMS.items():
        if acronym in words:
            words.update(expansion)
    return words - STOPWORDS


def _fallback_answer(question: str, records):
    """Return a corpus-grounded answer only when the topic clearly matches."""
    q_words = _words(question)
    if not q_words:
        return None

    best_answer = None
    best_score = 0.0
    for known_q, answer in records:
        k_words = _words(known_q)
        if not k_words:
            continue
        overlap = q_words & k_words
        # Prefer an exact normalized question match.
        if question.strip().lower().rstrip("?") == known_q.lower().rstrip("?"):
            return answer
        # Require at least one meaningful topic word, not just generic question words.
        if not overlap:
            continue
        score = len(overlap) / max(1, len(q_words))
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer is not None and best_score >= 0.50:
        return best_answer
    return None


def load_model(root: Path):
    vocab_path = root / "data" / "vocab.json"
    model_path = root / "models" / "transformer.pt"
    tokenizer = CharTokenizer.load(vocab_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = CyberGuardTransformer(
        vocab_size=checkpoint["vocab_size"],
        context_length=cfg.get("context_length", 128),
        embedding_dim=cfg.get("embedding_dim", 128),
        number_of_heads=cfg.get("heads", 4),
        number_of_layers=cfg.get("layers", 3),
        ffn_dim=cfg.get("ffn_dim", 256),
        dropout=cfg.get("dropout", 0.1),
        pad_id=checkpoint.get("pad_id", tokenizer.pad_id),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return tokenizer, model


def _model_answer(question: str, tokenizer, model, max_new_tokens=180) -> str:
    prompt = f"Question: {question.strip()}\nAnswer:"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.35, top_k=8)
    text = tokenizer.decode(generated[0].tolist())
    answer = text.split("Answer:", 1)[-1]
    answer = answer.split("\nQuestion:", 1)[0].strip()
    # Reject obviously corrupted generations rather than presenting gibberish.
    letters = re.findall(r"[A-Za-z]", answer)
    words = re.findall(r"[A-Za-z]+", answer)
    if len(letters) < 25 or len(words) < 5:
        return ""
    if any(answer.lower().count(fragment) >= 3 for fragment in ("cinty", "pr:", "ater:", "sse", "securit")):
        return ""
    return answer


def generate_answer(question: str, max_new_tokens=180) -> str:
    root = Path(__file__).resolve().parent
    records = _load_records(root)

    # First provide a grounded answer for known awareness topics.
    fallback = _fallback_answer(question, records)
    if fallback:
        return fallback

    # Otherwise let the trained Transformer attempt next-character generation.
    tokenizer, model = load_model(root)
    answer = _model_answer(question, tokenizer, model, max_new_tokens)
    if answer:
        return answer
    return ("I am not confident about that question. Please verify the information "
            "using a trusted cybersecurity source. Try asking about phishing, "
            "malware, passwords, MFA, social engineering, or safe browsing.")


if __name__ == "__main__":
    q = input("Ask a cybersecurity awareness question: ").strip()
    print(generate_answer(q))
