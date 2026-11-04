# train.py
"""Train the transformer LM on a body of text (and run a demo).

Usage:
    python train.py --data corpus.txt --epochs 20 --steps 2000

Without --data, it trains on a tiny built-in corpus so you can try it in seconds.
Saves a full checkpoint including the vocabulary to models/transfomer.pt.
"""

import argparse
import os
import random

import torch

from model import TransformerLM, Vocab, loss_fn, PAD

torch.manual_seed(42)
random.seed(42)

# Tiny built-in corpus (only used when --data is omitted).
DEFAULT_CORPUS = [
    "the cat sat on the mat and looked at the dog",
    "the dog ran in the yard and chased the cat",
    "i like to read books and write code",
    "the sun is shining and the birds are singing",
    "she likes to sing and dance in the garden",
    "the moon is bright tonight over the hill",
    "we went to the market and bought some flowers",
    "the train left the station at noon",
    "i work on ai projects and build web apps",
    "our team is learning natural language processing",
    "a neural network learns from data to make predictions",
    "attention lets the model focus on the important parts of a sentence",
]


def build_batch(ids_list, max_len, pad_idx=PAD):
    b = len(ids_list)
    x = torch.full((b, max_len), pad_idx, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        n = min(len(ids), max_len - 1)
        x[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
    return x


def prepare(corpus, max_len, pad_idx=PAD):
    vocab = Vocab().fit(corpus)
    samples = [vocab.encode(t) for t in corpus]
    return vocab, samples


def train(values, data_path=None):
    corpus = list(values["corpus"])
    if data_path:
        corpus += open(data_path, encoding="utf-8").read().splitlines()

    vocab, samples = prepare(corpus, values["max_len"])
    model = TransformerLM(
        vocab.vocab_size,
        d_model=values["d_model"],
        nhead=values["nhead"],
        d_ff=values["d_ff"],
        n_layers=values["n_layers"],
        max_len=values["max_len"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=values["lr"], weight_decay=1e-5)
    global_step = 0
    epochs = values["epochs"]
    steps = values["steps"]
    log_every = values["log_every"]
    batch_size = values.get("batch_size", 8)
    max_len = values.get("max_len", 64)

    print(f"[train] vocab_size={vocab.vocab_size} samples={len(samples)}")

    for _ in range(epochs):
        random.shuffle(samples)
        for start in range(0, len(samples), batch_size):
            if global_step >= steps:
                return
            batch = samples[start:start + batch_size]
            x = build_batch(batch, max_len, pad_idx=PAD)
            # y = shifted input; the final position is ignored (targets -100).
            y = torch.roll(x, shifts=-1, dims=-1)
            y[:, -1] = -100

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % log_every == 0:
                print(f"step {global_step}/{steps} loss={loss.item():.4f}")
            global_step += 1

    # keep the vocabulary on the model so a full checkpoint is reloadable
    model.vocab = vocab
    os.makedirs("models", exist_ok=True)
    torch.save(model, "models/transformer.pt")
    print(f"[train] saved models/transformer.pt (final loss={loss.item():.4f})")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="optional .txt corpus to train on")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    return p


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    args = build_parser().parse_args()
    corpus = []
    if args.data:
        corpus = open(args.data, encoding="utf-8").read().splitlines()
    else:
        corpus = list(DEFAULT_CORPUS)
    train(dict(
        corpus=corpus,
        epochs=args.epochs,
        steps=args.steps,
        log_every=args.log_every,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_len=args.max_len,
        lr=args.lr,
        batch_size=args.batch_size,
    ))
