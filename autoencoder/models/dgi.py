# -*- coding: utf-8 -*-
# models/dgi.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class AvgReadout(nn.Module):
    """全局读出：对节点嵌入做均值 + Tanh，用作摘要向量 s."""

    def __init__(self):
        super().__init__()
        # 移除了 apply_sigmoid 参数

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [N, D]
        s = h.mean(dim=0, keepdim=True)  # [1, D]
        # 直接使用 Tanh 保持数值稳定
        return torch.tanh(s)


class Discriminator(nn.Module):
    """双线性判别器： score_i = h_i^T W s"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # h: [N, D], s: [1, D] (broadcast)
        ws = torch.matmul(s, self.weight)  # [1, D]

        # 建议优化：加上 scaling 防止点积过大导致梯度爆炸
        scaling = h.size(1) ** -0.5
        scores = (h * ws).sum(dim=1) * scaling
        return scores


class DGI(nn.Module):
    """
    Deep Graph Infomax
    """

    def __init__(
            self,
            hidden_dim: int,
            corruption: Literal["row_shuffle", "col_shuffle", "gaussian", "permute"] = "row_shuffle",
            loss_kind: Literal["bce", "jsd"] = "bce",
            gaussian_std: float = 0.2
    ):
        super().__init__()
        # ================== 修改点在这里 ==================
        self.readout = AvgReadout()  # 不再传递 apply_sigmoid=True
        # ================================================

        self.disc = Discriminator(hidden_dim)
        self.corrupt_mode = corruption
        self.loss_kind = loss_kind
        self.gaussian_std = gaussian_std
        self.bce = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def corruption(self, x: torch.Tensor) -> torch.Tensor:
        """
        对原始特征做扰动
        """
        N, F = x.shape
        if self.corrupt_mode == "row_shuffle":
            idx = torch.randperm(N, device=x.device)
            return x[idx]
        elif self.corrupt_mode == "col_shuffle":
            idx = torch.randperm(F, device=x.device)
            return x[:, idx]
        elif self.corrupt_mode == "gaussian":
            return x + torch.randn_like(x) * self.gaussian_std
        else:  # "permute"
            return x[torch.randperm(N, device=x.device)]

    def loss(self, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
        """
        h_pos/h_neg 由外部 encoder 得到。
        """
        # 摘要向量 s 基于正样本
        s = self.readout(h_pos)  # [1, D]

        # 计算判别分数
        pos_logits = self.disc(h_pos, s)  # [N]
        neg_logits = self.disc(h_neg, s)  # [N]

        if self.loss_kind == "bce":
            y_pos = torch.ones_like(pos_logits)
            y_neg = torch.zeros_like(neg_logits)
            loss = self.bce(pos_logits, y_pos) + self.bce(neg_logits, y_neg)
            return loss
        else:  # "jsd" Jensen-Shannon 风格
            pos = F.logsigmoid(pos_logits).mean()
            neg = F.logsigmoid(-neg_logits).mean()
            return -(pos + neg)