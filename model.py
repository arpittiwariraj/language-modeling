# model.py
"""Transformer language model (PyTorch), self-supervised next-token prediction.

Forward:  x (batch, seq_len) -> logits (batch, seq_len, vocab)
Loss:     cross-entropy over shifted target tokens
Generate: feed a seed of token IDs and take greedy next-token

Multi-head attention via nn.TransformerEncoderLayer (batch_first=True) with a
causal mask, so gradients come from autograd (correct by construction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1


# ---------------- vocabulary ----------------
class Vocab:
    """Word-level vocabulary. Maps word<->id."""

    def __init__(self):
        self.stoi = {"<pad>": PAD, "<unk>": UNK}
        self.itos = {PAD: "<pad>", UNK: "<unk>"}

    def add(self, word):
        if word not in self.stoi:
            self.stoi[word] = len(self.stoi)
            self.itos[len(self.stoi)] = word
            return self.stoi[word]
        return self.stoi[word]

    def fit(self, texts):
        for t in texts:
            for w in t.split():
                self.add(w)
        return self

    def encode(self, text):
        return [self.stoi.get(w, UNK) for w in text.split()]

    def decode(self, ids):
        ids = [i for i in ids if i != PAD]
        return " ".join(self.itos.get(i, "") for i in ids if i >= 0)

    @property
    def vocab_size(self):
        return len(self.itos)


class TransformerLM(nn.Module):
    """Compact but complete transformer language model."""

    def __init__(self, vocab_size, d_model=128, nhead=4, d_ff=256,
                 n_layers=2, max_len=64, pad_idx=PAD):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, s):
        return torch.triu(torch.full((s, s), float("-inf")), diagonal=1)

    def forward(self, src):
        b, s = src.shape
        x = self.tok_emb(src) + self.pos_emb(torch.arange(s, device=src.device))
        mask = self._causal_mask(s) if self.training else None
        for blk in self.blocks:
            x = blk(x, src_mask=mask)
        return self.out(x)

    def generate(self, seed_ids, max_new=30):
        """Autoregressive next-token generation from a seed list of token IDs."""
        self.eval()
        ids = list(seed_ids)
        with torch.no_grad():
            for _ in range(max_new):
                if not ids:
                    break
                inp = torch.tensor([ids[-self.max_len:]], device=next(self.parameters()).device)
                logits = self(inp)[0, -1]
                nxt = int(logits.argmax())
                ids.append(nxt)
                if nxt == PAD:
                    break
        return ids


def loss_fn(logits, targets, ignore_index=-100):
    b, s, v = logits.shape
    return F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1),
                           ignore_index=ignore_index, reduction="mean")
