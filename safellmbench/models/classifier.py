"""
Custom Transformer safety classifier — architecture must match the checkpoint
shipped in `classifier_bundle.zip` (trained on allenai/wildguardmix).

The class definitions here are byte-compatible with the ones used at training
time in `notebooks/03_safety_classifier.ipynb` — do NOT rename fields or
change padding_idx, otherwise `load_state_dict` will fail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer as HfTokenizer

from .. import config


# ---------------------------------------------------------------------------
# Architecture (identical to notebook 03_safety_classifier.ipynb)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        Q = self.W_q(q).view(b, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(b, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(b, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = (attn @ V).transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.W_o(ctx), attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        a, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = self.dropout(self.embedding(x))
        x = self.pos_enc(x)
        for blk in self.encoder_blocks:
            x = blk(x, mask)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Loader + inference wrapper
# ---------------------------------------------------------------------------
@dataclass
class SafetyClassifier:
    model: TransformerClassifier
    tokenizer: HfTokenizer
    max_len: int
    device: str

    @torch.no_grad()
    def score(self, text: str) -> Tuple[bool, float]:
        """Return (is_harmful, probability) for one text."""
        if not text or not text.strip():
            return False, 0.0
        ids = self.tokenizer.encode(text).ids
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = (x == 0).unsqueeze(1).unsqueeze(2)
        prob = torch.sigmoid(self.model(x, mask)).item()
        return prob >= config.CLASSIFIER_THRESHOLD, round(prob, 4)


def load_classifier(device: str | None = None) -> SafetyClassifier:
    """Load the safety classifier from the local `~/.safellmbench/` install."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path: Path = config.CLASSIFIER_CKPT
    tok_path: Path = config.BPE_TOKENIZER_JSON
    if not ckpt_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            "Classifier bundle not installed. Run `safellmbench setup` first."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TransformerClassifier(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        max_len=cfg["max_len"],
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = HfTokenizer.from_file(str(tok_path))
    tok.enable_padding(length=cfg["max_len"], pad_id=0, pad_token="<PAD>")
    tok.enable_truncation(max_length=cfg["max_len"])

    return SafetyClassifier(model=model, tokenizer=tok,
                            max_len=cfg["max_len"], device=device)
