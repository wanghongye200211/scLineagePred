# -*- coding: utf-8 -*-
"""
训练脚本：支持 .h5 与 .h5ad，支持 DGI 先预训练与两层 MLP / GRN 解码器。
默认设备自动选择 CUDA -> MPS -> CPU。
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from dataio import load_h5_to_matrix, load_h5ad
from utils import (
    log1p_if_needed,
    minmax_0_1,
    read_prior_network,
    build_in_out_edges,
    build_coexp_knn,
)
from netmodel import CEFCON_AE_DGI, ModelCfg  # 与 netmodel.py 同级


@dataclass
class TrainConfig:
    # 数据
    expr_h5: Optional[str] = None
    expr_h5ad: Optional[str] = None
    gene_names_txt: Optional[str] = None
    h5_key: Optional[str] = None

    # 先验网络
    net_tsv: Optional[str] = None

    # 预处理
    do_log1p: bool = True

    # 模型结构与优化
    hidden: int = 256
    emb_dim: int = 128
    heads: int = 4
    tau: float = 0.25
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 256

    # 训练日程（默认：只做 DGI 预训练 + 联合）
    epochs_pretrain_dgi: int = 200
    early_stopping_delta: float = 5e-3

    epochs_pretrain: int = 0
    epochs_joint: int = 400
    early_stopping_patience: int = 80
    lambda_rec: float = 1.0
    lambda_dgi: float = 0.3
    verbose_every: int = 10

    # 输出
    out_dir: str = "./outputs_cefcon_ae_dgi"

    # gene-feature 宽度限速（可下采样细胞数以加速 GNN 编码器）
    max_cells_for_gene_features: Optional[int] = 5000

    # 解码与损失
    recon_space: str = "log"             # "log" | "count"
    rec_loss_kind: str = "mse_weighted"  # "mse" | "mse_weighted"

    # 解码器选择与结构
    decoder_kind: str = "grn"            # "mlp" | "symmetric" | "grn"
    dec_hidden1: int = 512
    dec_hidden2: int = 256
    dec_activation: str = "gelu"
    dec_dropout: float = 0.1


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) 读取 cells×genes
    if cfg.expr_h5 is not None:
        X_cg, genes = load_h5_to_matrix(
            cfg.expr_h5, gene_names_txt=cfg.gene_names_txt, h5_key=cfg.h5_key
        )
    elif cfg.expr_h5ad is not None:
        X_cg, genes = load_h5ad(cfg.expr_h5ad)
    else:
        raise ValueError("请提供 expr_h5 或 expr_h5ad 路径。")
    print(f"[data] X_cg: {X_cg.shape}, genes: {len(genes)}")

    # 2) 预处理
    X_cg = log1p_if_needed(X_cg, do_log1p=cfg.do_log1p).astype(np.float32)

    # 3) 基因节点特征 X_gf（仅用于 GNN 编码器）
    C, G = X_cg.shape
    if (cfg.max_cells_for_gene_features is not None) and (C > cfg.max_cells_for_gene_features):
        rng = np.random.RandomState(42)
        sel = rng.choice(C, size=cfg.max_cells_for_gene_features, replace=False)
        X_gf = X_cg[sel, :].T.copy()
        print(f"[feat] 使用下采样细胞 {len(sel)} 作为基因特征维度: X_gf={X_gf.shape} (原 F={C})")
    else:
        X_gf = X_cg.T.copy()
        print(f"[feat] 使用全部细胞作为基因特征: X_gf={X_gf.shape}")

    # 4) 构图：优先先验网络，否则共表达 KNN
    if cfg.net_tsv and os.path.exists(cfg.net_tsv):
        src, dst = read_prior_network(cfg.net_tsv, genes)
        print(f"[net] 使用先验网络：E={len(src)}")
    else:
        src, dst = build_coexp_knn(X_cg, topk=10)
        print(f"[net] 无先验网络，使用共表达 KNN：E={len(src)}")
    edge_in, edge_out = build_in_out_edges(src, dst)

    # 5) 注意力门控向量（例：跨细胞 std -> 0..1）
    #    这里不再依赖 log2FC 或伪时序，完全由数据本身的变异度决定
    l_vec = minmax_0_1(X_cg.std(axis=0)).astype(np.float32)

    # 6) 组网与训练
    model_cfg = ModelCfg(
        in_dim=X_gf.shape[1],
        hidden=cfg.hidden,
        out_dim=cfg.emb_dim,
        heads=cfg.heads,
        tau=cfg.tau,
        dropout=cfg.dropout,
        early_stopping_delta=cfg.early_stopping_delta,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        epochs_pretrain_dgi=cfg.epochs_pretrain_dgi,
        epochs_pretrain=cfg.epochs_pretrain,
        epochs_joint=cfg.epochs_joint,
        early_stopping_patience=cfg.early_stopping_patience,
        lambda_rec=cfg.lambda_rec,
        lambda_dgi=cfg.lambda_dgi,
        device=pick_device(),
        verbose_every=cfg.verbose_every,
        recon_space=cfg.recon_space,
        rec_loss_kind=cfg.rec_loss_kind,
        encoder_grad_agg="mean",
        decoder_kind=cfg.decoder_kind,
        dec_hidden1=cfg.dec_hidden1,
        dec_hidden2=cfg.dec_hidden2,
        dec_activation=cfg.dec_activation,
        dec_dropout=cfg.dec_dropout,
    )

    trainer = CEFCON_AE_DGI(model_cfg, num_genes=G)
    E_final, Z_cells = trainer.fit(
        X_cg=X_cg,
        X_gf=X_gf,
        edge_in=edge_in,
        edge_out=edge_out,
        l_vec=l_vec,
        out_dir=cfg.out_dir,
    )

    # 可选：输出 genes.txt
    with open(os.path.join(cfg.out_dir, "genes.txt"), "w", encoding="utf-8") as f:
        for g in genes:
            f.write(str(g) + "\n")

    print(f"[OK] Done. 输出目录: {cfg.out_dir}")
    print(" - Z_genes.npy（基因嵌入 E）")
    print(" - Z_cells.npy（细胞 latent）")
    print(" - model.pt / model_config.json /（若对称解码器）decoder_bias.npy")


if __name__ == "__main__":
    raise SystemExit(
        "Use `python -m sclineagepred embedding train ...` or import TrainConfig/main directly."
    )
