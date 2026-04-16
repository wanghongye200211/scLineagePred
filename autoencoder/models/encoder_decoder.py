# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ====================== 现有内容 ======================
# 你原有的三个模块：CellProjector / SymmetricDecoder / MLPDecoder
# 保持类名与接口不变，只改内部实现
# ----------------------------------------------------


class CellProjector(nn.Module):
    """X[B,G] 与 E[G,D] -> Z[B,D]；若 space='log' 则先近似到 count 再按行归一。"""
    def __init__(self, space: str = "log"):
        super().__init__()
        assert space in ("log", "count")
        self.space = space

    def forward(self, X: torch.Tensor, E: torch.Tensor):
        if self.space == "log":
            X_cnt = torch.expm1(torch.clamp(X, min=0.0))
        else:
            X_cnt = X
        rowsum = X_cnt.sum(dim=1, keepdim=True).clamp_min(1e-6)
        Z = (X_cnt / rowsum) @ E
        return Z, rowsum


class SymmetricDecoder(nn.Module):
    """
    与 E 权重对称：Xhat = Z @ E^T + b（log 域）
          或   rowsum * softplus(Z @ E^T + b)（count 域）
    """
    def __init__(self, num_genes: int, recon_space: str = "log", use_softplus: bool = True):
        super().__init__()
        assert recon_space in ("log", "count")
        self.recon_space = recon_space
        self.use_softplus = use_softplus
        self.bias = nn.Parameter(torch.zeros(num_genes, dtype=torch.float32))

    def forward(self, Z: torch.Tensor, E: torch.Tensor, rowsum=None):
        logits = Z @ E.t() + self.bias
        if self.recon_space == "count":
            assert rowsum is not None, "count 域解码需要 rowsum"
            pred = F.softplus(logits) if self.use_softplus else torch.relu(logits)
            return rowsum * pred
        else:
            return logits

    @torch.no_grad()
    def decode_from_latent(self, Z: torch.Tensor, E: torch.Tensor, rowsum=None):
        return self.forward(Z, E, rowsum)


class MLPDecoder(nn.Module):
    """
    两层 MLP 解码器：Z[B,D] --(H1)-> --(H2)-> X_hat[B,G]
    - recon_space='log': 直接输出 logits
    - recon_space='count': 输出 softplus(logits) * rowsum
    设计成与 SymmetricDecoder 相同的前向签名，便于在 netmodel 中无缝切换。
    """
    def __init__(
        self,
        in_dim: int,
        num_genes: int,
        hidden1: int,
        hidden2: int,
        recon_space: str = "log",
        act: str = "gelu",
        dropout: float = 0.0,
        use_softplus: bool = True,
    ):
        super().__init__()
        assert recon_space in ("log", "count")
        self.recon_space = recon_space
        self.use_softplus = use_softplus

        self.lin1 = nn.Linear(in_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.out = nn.Linear(hidden2, num_genes)
        self.drop = nn.Dropout(dropout)

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.GELU()

    def forward(self, Z: torch.Tensor, E_ignored: torch.Tensor = None, rowsum=None):
        h = self.lin1(Z)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.lin2(h)
        h = self.bn2(h)
        h = self.act(h)

        logits = self.out(h)
        if self.recon_space == "count":
            assert rowsum is not None, "count 域解码需要 rowsum"
            pred = F.softplus(logits) if self.use_softplus else torch.relu(logits)
            return rowsum * pred
        else:
            return logits

    @torch.no_grad()
    def decode_from_latent(self, Z: torch.Tensor, E_ignored: torch.Tensor = None, rowsum=None):
        return self.forward(Z, None, rowsum)


# ====================== 新增内容：GRN 对称解码器 ======================
# 1) 用与你的编码器同构的两层双向注意力，对 E[G,D] 做“图读出”，得到 E_tilde[G,D]
# 2) 然后按对称解码公式 Xhat = Z @ E_tilde^T + b
# 3) 自带图缓存：一个 epoch 的重构循环中多次调用 forward 时，只要 E 指针没变，就不重复图读出计算

from models.layers import _BiDirectionalBlock  # noqa: E402


class GraphSymmetricDecoder(nn.Module):
    """
    GRN 对称解码器：
      - 结构对称：两层 _BiDirectionalBlock（与编码器同构），输入是基因嵌入 E[G,D]，
                  输出为 E_tilde[G,D]（已经融合了网络信息）
      - 解码公式：Xhat = Z[B,D] @ E_tilde[G,D]^T + b[G]
      - recon_space='log' | 'count' 与 SymmetricDecoder 完全一致
      - attach_graph(): 绑定 edge_in/edge_out/l_vec；forward() 内部自动使用
      - clear_cache(): 清理 E_tilde 缓存（建议在每个 AE 重构 epoch 开始时调用一次）
    """
    def __init__(
        self,
        num_genes: int,
        emb_dim: int,
        hidden: int,
        heads: int = 4,
        tau: float = 0.25,
        dropout: float = 0.1,
        recon_space: str = "log",
        use_softplus: bool = True,
    ):
        super().__init__()
        assert recon_space in ("log", "count")
        assert hidden % 2 == 0, "GRN 解码器的 hidden 需为偶数（双向拼接）"

        self.recon_space = recon_space
        self.use_softplus = use_softplus
        self.num_genes = num_genes
        self.emb_dim = emb_dim

        # 两层与 encoder 同构的双向注意力块（D -> hidden -> D）
        self.read1 = _BiDirectionalBlock(in_dim=emb_dim, out_dim=hidden, heads=heads, tau=tau, dropout=dropout)
        self.read2 = _BiDirectionalBlock(in_dim=hidden, out_dim=emb_dim, heads=heads, tau=tau, dropout=dropout)

        # 基因偏置（保持与 SymmetricDecoder 一致）
        self.bias = nn.Parameter(torch.zeros(num_genes, dtype=torch.float32))

        # 图结构缓存在 buffer 中
        self.register_buffer("_edge_in", None, persistent=False)
        self.register_buffer("_edge_out", None, persistent=False)
        self.register_buffer("_l_vec", None, persistent=False)

        # E_tilde 缓存（按 data_ptr 区分）
        self._cache_E_ptr = None
        self._cache_E_tilde = None

    def attach_graph(
        self,
        edge_in: torch.Tensor,
        edge_out: torch.Tensor,
        l_vec: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        绑定基因网络与门控向量，便于在 forward 内部直接使用。
        参数均为 torch.Tensor（edge:[2,E] Long；l_vec:[G] Float），会自动搬到 decoder 的 device 上。
        """
        dev = device if device is not None else (
            self.bias.device if hasattr(self, "bias") else torch.device("cpu")
        )
        self._edge_in = edge_in.to(dev).long()
        self._edge_out = edge_out.to(dev).long()
        self._l_vec = None if l_vec is None else l_vec.to(dev)
        self.clear_cache()

    def clear_cache(self):
        """清空 E_tilde 缓存；建议在每个 AE 重构 epoch 开始时调用一次。"""
        self._cache_E_ptr = None
        self._cache_E_tilde = None

    def _graph_readout(self, E: torch.Tensor) -> torch.Tensor:
        """
        对 E[G,D] 做两层 GRN 读出，得到 E_tilde[G,D]。内部带缓存。
        """
        assert self._edge_in is not None and self._edge_out is not None, "GraphSymmetricDecoder: 未 attach_graph()"
        cur_ptr = int(E.data_ptr())

        # 若 E 的 data_ptr 未变化，直接复用缓存
        if (self._cache_E_ptr == cur_ptr) and (self._cache_E_tilde is not None):
            return self._cache_E_tilde

        H1 = self.read1(E, self._edge_in, self._edge_out, self._l_vec)  # [G, hidden]
        H2 = self.read2(H1, self._edge_in, self._edge_out, self._l_vec)  # [G, D]

        self._cache_E_ptr = cur_ptr
        self._cache_E_tilde = H2
        return H2

    def forward(self, Z: torch.Tensor, E: torch.Tensor, rowsum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z: [B, D], E: [G, D]  ->  Xhat: [B, G]
        recon_space="log":    logits = Z @ E_tilde^T + b
        recon_space="count":  rowsum * softplus(logits)
        """
        E_tilde = self._graph_readout(E)
        logits = Z @ E_tilde.t() + self.bias

        if self.recon_space == "count":
            assert rowsum is not None, "count 域解码需要 rowsum"
            pred = F.softplus(logits) if self.use_softplus else torch.relu(logits)
            return rowsum * pred
        else:
            return logits

    @torch.no_grad()
    def decode_from_latent(self, Z: torch.Tensor, E: torch.Tensor, rowsum: Optional[torch.Tensor] = None):
        return self.forward(Z, E, rowsum)
