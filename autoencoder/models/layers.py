# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class GraphAttention_layer(MessagePassing):
    """
    多头绝对值余弦注意力 + 节点级门控：
      - attention_ij ~ |cos(h_i, h_j)| * sigmoid(c * l_j + d)
      - l_j 来自基因的一个标量特征（这里由外部传入 l_vec）
    为兼容 Apple MPS，segment softmax 用 index_add_ 自己实现。

    输入:
      x: [N, Fin]
      edge_index: [2, E] (src->dst)
      l_vec: [N] or None
    输出:
      out: [N, Fout]
      att (可选): [E, heads]
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, tau: float = 0.25, dropout: float = 0.0):
        super().__init__(aggr="add", node_dim=0)
        assert out_dim % heads == 0, "GraphAttention_layer: out_dim 必须能被 heads 整除"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dh = out_dim // heads
        self.tau = tau

        self.dropout = nn.Dropout(dropout)

        # 多头 Q / KV
        self.W_q = nn.Parameter(torch.randn(heads, in_dim, self.dh) * (1.0 / math.sqrt(in_dim)))
        self.W_kv = nn.Parameter(torch.randn(heads, in_dim, self.dh) * (1.0 / math.sqrt(in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # 节点级门控 D_j = sigmoid(c * l_j + d)
        self.c = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(0.0))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_kv)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.c.fill_(1.0)
            self.d.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,                # [N, Fin]
        edge_index: torch.Tensor,       # [2, E]
        l_vec: Optional[torch.Tensor] = None,
        return_att: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        N = x.size(0)

        # 线性映射 -> [N, H, Dh]
        xq = torch.einsum("nf,hfd->nhd", x, self.W_q)
        xkv = torch.einsum("nf,hfd->nhd", x, self.W_kv)

        eps = 1e-9
        nq = torch.linalg.norm(xq, dim=-1, keepdim=True) + eps
        nkv = torch.linalg.norm(xkv, dim=-1, keepdim=True) + eps

        if l_vec is None:
            l_nodes = x.new_ones(N)
        else:
            l_nodes = l_vec.view(-1)
            if l_nodes.numel() != N:
                raise ValueError(f"l_vec 长度({l_nodes.numel()})应等于节点数 N({N})")

        out = self.propagate(
            edge_index=edge_index,
            x=(xq, xkv),            # -> x_i, x_j
            x_norm=(nq, nkv),       # -> x_norm_i, x_norm_j
            x_auxiliary=l_nodes,    # -> x_auxiliary_i, x_auxiliary_j
            size=(N, N),
        )                           # [N, H, Dh]

        out = out.reshape(N, self.out_dim) + self.bias
        if return_att and hasattr(self, "_last_alpha"):
            return out, self._last_alpha
        return out, None

    def _segment_softmax(self, score: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        按目标节点 index[E] 对 score[E, H] 做分组 softmax。
        """
        tau = max(self.tau, 1e-6)
        exp_s = torch.exp(score / tau)  # [E, H]

        denom = torch.zeros(num_nodes, exp_s.size(1), device=exp_s.device, dtype=exp_s.dtype)  # [N, H]
        denom.index_add_(0, index, exp_s)
        denom_e = denom.index_select(0, index)  # [E, H]

        alpha = exp_s / (denom_e + 1e-9)
        return alpha

    def message(
        self,
        x_i: torch.Tensor,             # [E, H, Dh] 目标
        x_j: torch.Tensor,             # [E, H, Dh] 源
        x_norm_i: torch.Tensor,        # [E, H, 1]
        x_norm_j: torch.Tensor,        # [E, H, 1]
        x_auxiliary_j: Optional[torch.Tensor],  # [E]
        edge_index_i: torch.Tensor,    # [E]
        size_i: Optional[int],
    ):
        # 绝对值余弦相似度
        dot = (x_i * x_j).sum(dim=-1)  # [E, H]
        cos = torch.abs(dot / (x_norm_i.squeeze(-1) * x_norm_j.squeeze(-1)))  # [E, H]

        # 节点级门控（例如用 l_j = 基因 std）
        if x_auxiliary_j is None:
            Dj = cos.new_ones(cos.size(0), 1)  # [E, 1]
        else:
            Dj = torch.sigmoid(self.c * x_auxiliary_j.view(-1, 1) + self.d)  # [E, 1]

        score = cos * Dj  # [E, H]
        alpha = self._segment_softmax(
            score,
            edge_index_i,
            num_nodes=size_i if size_i is not None else int(x_i.size(0)),
        )
        alpha = self.dropout(alpha)
        self._last_alpha = alpha
        return alpha.unsqueeze(-1) * x_j  # [E, H, Dh]


# -------------------- 双向两层编码器 --------------------
class _BiDirectionalBlock(nn.Module):
    """
    一层“入边 + 出边”双向 GAT + 残差 MLP
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int, tau: float, dropout: float):
        super().__init__()
        assert out_dim % 2 == 0, "BiDirectionalBlock: out_dim 需为偶数（in/out 各一半）"
        half = out_dim // 2

        self.gat_in = GraphAttention_layer(in_dim, half, heads, tau, dropout)
        self.gat_out = GraphAttention_layer(in_dim, half, heads, tau, dropout)

        self.lin = nn.Linear(out_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_in, edge_out, l_vec):
        # edge_in: src->dst
        # edge_out: dst->src（反向图）
        h_in, _ = self.gat_in(x, edge_in, l_vec, return_att=False)
        h_out, _ = self.gat_out(x, edge_out, l_vec, return_att=False)
        h = torch.cat([h_in, h_out], dim=1)
        h = self.lin(h)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class GRNEncoder(nn.Module):
    """
    两层双向注意力编码器：
      输入: X_gf[G, F]（基因节点特征）
      输出: E[G, D]（基因嵌入）
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, heads: int = 4, tau: float = 0.25, dropout: float = 0.2):
        super().__init__()
        self.l1 = _BiDirectionalBlock(in_dim, hidden, heads, tau, dropout)
        self.l2 = _BiDirectionalBlock(hidden, out_dim, heads, tau, dropout)

    def forward(self, x, edge_in, edge_out, l_vec):
        h = self.l1(x, edge_in, edge_out, l_vec)
        h = self.l2(h, edge_in, edge_out, l_vec)
        return h
