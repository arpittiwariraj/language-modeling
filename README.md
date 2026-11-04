# Language Modeling Transformer

A compact, runnable **transformer language model** implemented in PyTorch. Given a body of text, it learns to predict the next token and then **generates** original text from a seed.

The model is a proper transformer: token + positional embedding, multi-head
self-attention with a causal mask, position-wise feed-forward, layer norm, and a
final vocabulary head. All gradients come from PyTorch's autograd, so it trains
correctly out of the box.

## Quick start
```bash
# 1) clone & enter
git clone https://github.com/arpittiwariraj/language-modeling.git
cd language-modeling

# 2) make a virtualenv & install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) train (built-in tiny corpus by default)
python train.py

# 4) generate text
python generate.py --seed "the cat" --length 40
```

Train with your own dataset:
```bash
python train.py --data my_corpus.txt --epochs 30 --steps 5000
```

## What's inside
```
language-modeling/
├── model.py        # TransformerLM + Vocab (pure PyTorch)
├── train.py        # Training loop + checkpoint saving
├── generate.py     # Text generation from a checkpoint
├── requirements.txt# deps (torch + numpy)
├── data/           # (optional) your own training text
└── README.md
```

## Notes
- The built-in corpus is tiny and only used to demonstrate the pipeline. Pass a
  real corpus with `--data` to train a useful model.
- Checkpoints are saved to `models/transfomer.pt` after training.
- This is a learning experiment — results will be small until you train on a
  larger corpus.

## License
MIT
