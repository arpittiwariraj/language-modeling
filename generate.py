# generate.py
"""Generate text from a trained checkpoint.

Creates a checkpoint including the vocabulary, so a full reload is possible.

Usage:
    python train.py            # train on the built-in corpus
    python generate.py --seed "the cat" --length 40
"""

import argparse
import os
import sys

import torch

# make sure this directory (with model.py / Vocab) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TransformerLM, PAD  # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "transformer.pt")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CKPT, help="path to the saved model (.pt)")
    p.add_argument("--seed", default="the cat")
    p.add_argument("--length", type=int, default=40)
    return p


def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"[gen] no checkpoint at {args.checkpoint}; run train.py first")
        return

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.eval()
    print(f"[gen] loaded {args.checkpoint}")

    seed = model.vocab.encode(args.seed)
    ids = model.generate(seed, max_new=args.length)
    print(f"  seed     : {args.seed}")
    print(f"  generated: {model.vocab.decode(ids)}")


if __name__ == "__main__":
    main()
