"""Train CyberGuard with assignment-aligned recommended configuration.

Default configuration:
- batch size 16
- context length 128
- embedding 128
- heads 4
- Transformer layers 3
- FFN hidden size 256
- dropout 0.1
- learning rate 3e-4
- 3000 optimization steps
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from tokenizer import CharTokenizer
from dataset import build_dataloaders
from transformer import CyberGuardTransformer


def evaluate(model, loader, device, max_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "A", "B"], default="baseline",
                        help="baseline=3x128; A=2 layers/128; B=4 layers/256")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()

    # Assignment experiments. These can be overridden with command-line flags.
    if args.experiment == "A":
        args.layers, args.embedding_dim, args.ffn_dim = 2, 128, 256
    elif args.experiment == "B":
        args.layers, args.embedding_dim, args.ffn_dim = 4, 256, 512

    torch.manual_seed(42)
    root = Path(__file__).resolve().parent
    corpus = root / "data" / "corpus.jsonl"
    vocab = root / "data" / "vocab.json"
    model_name = "transformer.pt" if args.experiment == "baseline" else f"experiment_{args.experiment}.pt"
    model_path = root / "models" / model_name
    report_path = root / "data" / "training_report.json"

    tokenizer = CharTokenizer.from_corpus(corpus)
    tokenizer.save(vocab)
    tokenizer, splits, loaders = build_dataloaders(corpus, vocab, args.batch_size, args.context_length)
    train_records, val_records, test_records = splits
    train_loader, val_loader, test_loader = loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberGuardTransformer(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        number_of_heads=args.heads,
        number_of_layers=args.layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()
    history = []
    train_iter = iter(train_loader)
    best_val = float("inf")

    model.train()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_loader, device)
            history.append({"step": step, "train_loss": loss.item(), "validation_loss": val_loss})
            print(f"Step {step:4d}/{args.steps} | train={loss.item():.4f} | val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "vocab_size": tokenizer.vocab_size,
                    "pad_id": tokenizer.pad_id,
                    "trained_steps": step,
                    "best_validation_loss": best_val,
                }, model_path)

    elapsed = time.time() - start_time
    # Test loss using the best saved model.
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device, max_batches=1000)

    report = {
        "domain": "Cybersecurity Awareness",
        "experiment": args.experiment,
        "tokenizer": "character-level",
        "records_total": len(train_records) + len(val_records) + len(test_records),
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "test_records": len(test_records),
        "split": "80/10/10",
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": round(elapsed, 2),
        "best_validation_loss": best_val,
        "test_loss": test_loss,
        "history": history,
        "config": vars(args),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Training complete.")
    print("Saved model:", model_path)
    print("Test loss:", round(test_loss, 4))


if __name__ == "__main__":
    main()
