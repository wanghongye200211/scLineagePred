from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def _masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor):
    if key_padding_mask is None:
        return x.mean(dim=1)
    valid = (~key_padding_mask).unsqueeze(-1).type_as(x)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def _is_suffix_padding_mask(key_padding_mask: torch.Tensor) -> bool:
    if key_padding_mask is None:
        return True
    valid = (~key_padding_mask).to(torch.int32)
    return bool(((valid[:, :-1] - valid[:, 1:]) >= 0).all().item())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :].to(x.dtype))


class DirectPredictor(nn.Module):
    def __init__(self, kind: str, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.kind = kind

        if kind == "RNN":
            self.net = nn.RNN(in_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
            enc_out = hidden
        elif kind == "BiLSTM":
            self.net = nn.LSTM(in_dim, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
            enc_out = hidden * 2
        elif kind == "Trans":
            d_model = 128
            self.proj = nn.Linear(in_dim, d_model)
            self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=512)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=512,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.net = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            enc_out = d_model
        else:
            raise ValueError(f"Unknown kind: {kind}")

        self.head = nn.Sequential(
            nn.Linear(enc_out, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        if self.kind == "Trans":
            hidden = self.proj(x)
            hidden = self.pos(hidden)
            hidden = self.net(hidden, src_key_padding_mask=key_padding_mask)
            return self.head(_masked_mean(hidden, key_padding_mask))

        if key_padding_mask is not None and _is_suffix_padding_mask(key_padding_mask):
            lengths = (~key_padding_mask).sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            if self.kind == "RNN":
                _, hidden = self.net(packed)
                features = hidden[-1]
            else:
                _, (hidden, _) = self.net(packed)
                features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            outputs, state = self.net(x)
            if self.kind == "RNN":
                features = _masked_mean(outputs, key_padding_mask)
            else:
                features = _masked_mean(outputs, key_padding_mask)

        return self.head(features)
