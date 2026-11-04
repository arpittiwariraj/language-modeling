# AGENTS.md

This repository contains a **self-contained transformer language model** written in
pure PyTorch. It learns to predict the next word and then generates original text.

## Repository layout
- `model.py`  - `Vocab` + `TransformerLM`; purely PyTorch (`torch>=2.0`, `numpy`).
- `train.py`  - sentiment-free training loop; saves a full checkpoint (`models/transformer.pt`).
- `generate.py` - generate text from a checkpoint.
- `README.md` - usage.
- `data/sample.txt` - optional tiny corpus to try training without a file.

## Run
```bash
pip install -r requirements.txt
python train.py            # default built-in corpus
python train.py --data data/sample.txt --epochs 8 --steps 2000
python generate.py --seed "the cat" --length 30
```

## Notes for contributors
- Dependency: `torch` (CPU is enough; no CUDA required) and `numpy`.
- No Hugging Face / TensorFlow / JAX needed.
- The `generate.py` checkpoint is a **full model object** (weights + vocab) via
  `torch.save(model, ...)`, so nothing extra has to be reconstructed.
- Vocabulary IDs: `0` = `<pad>`, `1` = ``, rest = words.

## Coding conventions
- `model.py`: single-file PyTorch model; no implicit device handling (CPU/CPU-only).
- `train.py`: keeps a tiny built-in corpus as a demo; `--data` loads an external file.
- Before committing, test the loop with `python train.py --epochs 1 --steps 300`
  and spot-check with `python generate.py --seed "the" --length 20`.
