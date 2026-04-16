# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import os
import json

import numpy as np
import torch
import torch.nn as nn

from models.layers import GRNEncoder
from models.dgi import DGI
from models.encoder_decoder import (
    CellProjector,
    SymmetricDecoder,
    MLPDecoder,
    GraphSymmetricDecoder,
)

__version__ = "2025-11-08-grn-symmetric-decoder"


@dataclass
class ModelCfg:
    in_dim: int
    hidden: int = 256
    out_dim: int = 128
    heads: int = 4
    tau: float = 0.25
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 256

    # 训练日程
    epochs_pretrain_dgi: int = 200
    epochs_pretrain: int = 0
    epochs_joint: int = 400

    early_stopping_patience: int = 30  # 如果 > 0 则启用早停
    early_stopping_delta: float = 1e-4
    # 损失权重
    lambda_rec: float = 1.0
    lambda_dgi: float = 0.3

    # 设备与日志
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    seed: int = 42
    verbose_every: int = 10

    # 解码与损失
    recon_space: str = "log"            # "log" or "count"
    rec_loss_kind: str = "mse_weighted" # "mse" or "mse_weighted"
    encoder_grad_agg: str = "mean"      # "mean" or "sum"

    # 解码器选择与结构
    decoder_kind: str = "grn"           # "mlp" | "symmetric" | "grn"
    dec_hidden1: int = 512
    dec_hidden2: int = 256
    dec_activation: str = "gelu"
    dec_dropout: float = 0.1

from utils import EarlyStopping
class CEFCON_AE_DGI:
    """
    训练流程（可选三阶段）：
      A) DGI 预训练：只更新 encoder + dgi
      B) AE 预训练：只做重构（decoder 每 step 更新；对 encoder 累计 dL/dE，epoch 末一次性反传）
      C) 联合训练：AE + DGI
    最终导出：
      - E_final[G,D]: 基因嵌入
      - Z_cells[C,D]: 细胞 latent summary
    """
    def __init__(self, cfg: ModelCfg, num_genes: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.dgi_norm = nn.LayerNorm(cfg.out_dim).to(self.device)

        # 编码器 / DGI
        self.encoder = GRNEncoder(
            cfg.in_dim,
            cfg.hidden,
            cfg.out_dim,
            heads=cfg.heads,
            tau=cfg.tau,
            dropout=cfg.dropout,
        ).to(self.device)
        self.dgi = DGI(cfg.out_dim).to(self.device)

        # Projector 与 解码器
        self.projector = CellProjector(space=cfg.recon_space)
        self.decoder = self._make_decoder(cfg, num_genes).to(self.device)

        # 优化器
        self.opt_enc = torch.optim.Adam(
            self.encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.opt_dgi = torch.optim.Adam(
            self.dgi.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.opt_dec = torch.optim.Adam(
            self.decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        self.rec_weight: Optional[torch.Tensor] = None
        self.num_genes = num_genes

        # 图结构占位（用于后续 KO / Transformer 等）
        self._edge_in: Optional[np.ndarray] = None
        self._edge_out: Optional[np.ndarray] = None
        self._l_vec: Optional[np.ndarray] = None

        print(
            f"[netmodel] {__version__} | device={self.device} | "
            f"recon_space={cfg.recon_space} | rec_loss={cfg.rec_loss_kind} | decoder={cfg.decoder_kind}"
        )
        print("[netmodel] stages: DGI-pretrain -> AE-pretrain -> Joint (all optional)")

    # ------------------------------------------------------------------
    #  解码器构造
    # ------------------------------------------------------------------
    def _make_decoder(self, cfg: ModelCfg, num_genes: int) -> nn.Module:
        kind = cfg.decoder_kind.lower()
        if kind == "mlp":
            return MLPDecoder(
                in_dim=cfg.out_dim,
                num_genes=num_genes,
                hidden1=cfg.dec_hidden1,
                hidden2=cfg.dec_hidden2,
                recon_space=cfg.recon_space,
                act=cfg.dec_activation,
                dropout=cfg.dec_dropout,
                use_softplus=True,
            )
        elif kind == "symmetric":
            return SymmetricDecoder(
                num_genes=num_genes,
                recon_space=cfg.recon_space,
                use_softplus=True,
            )
        elif kind == "grn":
            # 这里 dec_hidden1 用作 GRN 解码器的隐藏宽度；需为偶数（双向拼接）
            hidden = max(
                2,
                cfg.dec_hidden1 if (cfg.dec_hidden1 % 2 == 0) else (cfg.dec_hidden1 + 1),
            )
            return GraphSymmetricDecoder(
                num_genes=num_genes,
                emb_dim=cfg.out_dim,
                hidden=hidden,
                heads=cfg.heads,
                tau=cfg.tau,
                dropout=cfg.dec_dropout,
                recon_space=cfg.recon_space,
                use_softplus=True,
            )
        else:
            raise ValueError(f"未知的 decoder_kind: {cfg.decoder_kind}")

    # ------------------------------------------------------------------
    #  基因编码
    # ------------------------------------------------------------------
    def _encode_genes(
        self,
        X_gf: np.ndarray,
        edge_in: np.ndarray,
        edge_out: np.ndarray,
        l_vec: Optional[np.ndarray],
    ) -> torch.Tensor:
        x = torch.from_numpy(X_gf).to(self.device)               # [G,F]
        ei = torch.from_numpy(edge_in).long().to(self.device)    # [2,E]
        eo = torch.from_numpy(edge_out).long().to(self.device)   # [2,E]
        lv = torch.from_numpy(l_vec).to(self.device) if l_vec is not None else None
        E = self.encoder(x, ei, eo, lv)                          # [G,D]
        return E

    # ------------------------------------------------------------------
    #  重构损失
    # ------------------------------------------------------------------
    def _rec_loss(self, Xhat: torch.Tensor, Xb: torch.Tensor) -> torch.Tensor:
        if self.cfg.rec_loss_kind == "mse_weighted":
            assert self.rec_weight is not None
            return torch.mean(((Xhat - Xb) ** 2) * self.rec_weight)
        return torch.mean((Xhat - Xb) ** 2)

    # ------------------------------------------------------------------
    #  AE 单 epoch：只更新 decoder，累计 dL/dE
    # ------------------------------------------------------------------
    def _iterate_recon(self, X_cg: np.ndarray, E_full: torch.Tensor):
        """
        X_cg: [C, G] numpy
        E_full: [G, D] torch
        返回:
          mean_loss, dL/dE_full (同形状)
        """
        self.encoder.train()
        self.decoder.train()

        bs = self.cfg.batch_size
        N = X_cg.shape[0]
        total_loss = 0.0
        steps = 0

        # 基于当前 encoder 产出的 E_full 作为“静态底座”，只更新 decoder。
        base_E = E_full.detach().clone()
        E_grad_sum = torch.zeros_like(base_E, device=self.device)

        # 如果是 GraphSymmetricDecoder，清一下内部缓存
        if isinstance(self.decoder, GraphSymmetricDecoder):
            self.decoder.clear_cache()

        for st in range(0, N, bs):
            ed = min(st + bs, N)
            Xb = torch.from_numpy(X_cg[st:ed]).to(self.device)

            # 每个 mini-batch 拷贝一份独立的叶子，形成独立计算图
            E_leaf = base_E.detach().clone().requires_grad_(True)

            Zb, rowsum = self.projector(Xb, E_leaf)
            Xhat = self.decoder(
                Zb,
                E_leaf,
                rowsum if self.cfg.recon_space == "count" else None,
            )
            loss = self._rec_loss(Xhat, Xb) * self.cfg.lambda_rec

            # 只更新 decoder；对 E_leaf 求梯度并累加
            self.opt_dec.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_dec.step()

            if E_leaf.grad is not None:
                E_grad_sum += E_leaf.grad.detach()

            total_loss += float(loss.item())
            steps += 1

            del E_leaf, Zb, Xhat, loss

        if self.cfg.encoder_grad_agg == "mean":
            E_grad_sum /= max(steps, 1)
        return (total_loss / max(steps, 1)), E_grad_sum

    # ------------------------------------------------------------------
    #  DGI 一个 step
    # ------------------------------------------------------------------
    def _step_dgi(
        self,
        X_gf: np.ndarray,
        edge_in: np.ndarray,
        edge_out: np.ndarray,
        l_vec: Optional[np.ndarray],
    ) -> float:
        x_pos = torch.from_numpy(X_gf).to(self.device)
        ei = torch.from_numpy(edge_in).long().to(self.device)
        eo = torch.from_numpy(edge_out).long().to(self.device)
        lv = torch.from_numpy(l_vec).to(self.device) if l_vec is not None else None


        E_pos = self.encoder(x_pos, ei, eo, lv)
        E_pos = self.dgi_norm(E_pos)
        with torch.no_grad():
            x_neg = self.dgi.corruption(x_pos)
        E_neg = self.encoder(x_neg, ei, eo, lv)
        E_neg = self.dgi_norm(E_neg)  # 负样本也要归一化

        dgi_loss = self.dgi.loss(E_pos, E_neg) * self.cfg.lambda_dgi
        self.opt_enc.zero_grad()
        self.opt_dgi.zero_grad()
        dgi_loss.backward()
        self.opt_enc.step()
        self.opt_dgi.step()
        return float(dgi_loss.item())

    # ------------------------------------------------------------------
    #  训练主流程
    # ------------------------------------------------------------------
    # models/netmodel.py

    # 记得在头部导入
    from utils import EarlyStopping

    # ... inside CEFCON_AE_DGI class ...

    def fit(
            self,
            X_cg: np.ndarray,
            X_gf: np.ndarray,
            edge_in: np.ndarray,
            edge_out: np.ndarray,
            l_vec: Optional[np.ndarray] = None,
            out_dir: str = "./results",
    ):
        os.makedirs(out_dir, exist_ok=True)
        # ===== [新增] 保存“严格 GRN 解码”所需的全部图信息与完整配置 =====
        # 1) 保存图结构（GRN decode 必需）
        np.save(os.path.join(out_dir, "edge_in.npy"),  edge_in.astype(np.int64))   # [2, E]
        np.save(os.path.join(out_dir, "edge_out.npy"), edge_out.astype(np.int64))  # [2, E]
        if l_vec is not None:
            np.save(os.path.join(out_dir, "l_vec.npy"), l_vec.astype(np.float32))  # [G]

        # 2) 保存完整 cfg（model_config.json 目前太简略，不利于跨工程复现）
        with open(os.path.join(out_dir, "model_cfg_full.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "netmodel_version": __version__,
                    "cfg": asdict(self.cfg),
                    "num_genes": int(self.num_genes),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 定义保存权重的闭包函数，方便复用
        model_save_path = os.path.join(out_dir, "model.pt")

        def save_current_model():
            torch.save(
                {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "dgi": self.dgi.state_dict(),
                    "cfg": asdict(self.cfg),
                },
                model_save_path,
            )
            # 保存 bias
            if hasattr(self.decoder, "bias"):
                np.save(
                    os.path.join(out_dir, "decoder_bias.npy"),
                    self.decoder.bias.detach().cpu().numpy().astype(np.float32),
                )

        # 保存图结构到成员
        self._edge_in, self._edge_out, self._l_vec = edge_in, edge_out, l_vec

        # 若是 GRN 解码器，绑一次图
        if isinstance(self.decoder, GraphSymmetricDecoder):
            ei_t = torch.from_numpy(edge_in).long().to(self.device)
            eo_t = torch.from_numpy(edge_out).long().to(self.device)
            lv_t = None if l_vec is None else torch.from_numpy(l_vec).to(self.device)
            self.decoder.attach_graph(ei_t, eo_t, lv_t, device=self.device)

        # MSE 权重计算 (保持不变)
        if self.cfg.rec_loss_kind == "mse_weighted":
            var = np.var(X_cg, axis=0).astype(np.float32) + 1e-6
            w = 1.0 / var
            w = w / w.mean()
            self.rec_weight = torch.from_numpy(w).to(self.device)

        # ===== A) DGI 预训练 (保持不变) =====
        if self.cfg.epochs_pretrain_dgi > 0:
            self.encoder.train()
            self.dgi.train()
            for epoch in range(1, self.cfg.epochs_pretrain_dgi + 1):
                dgi_loss = self._step_dgi(X_gf, edge_in, edge_out, l_vec)
                if epoch % self.cfg.verbose_every == 0 or epoch in (1, self.cfg.epochs_pretrain_dgi):
                    print(f"[Pretrain-DGI] epoch {epoch:4d}/{self.cfg.epochs_pretrain_dgi}  dgi={dgi_loss:.4f}")

        # ===== B) AE 预训练：加入早停（监控 recon） =====
        if self.cfg.epochs_pretrain > 0:
            early_stopper_ae = None
            if self.cfg.early_stopping_patience > 0:
                print(
                    f"[Pretrain-AE ] 启用早停: patience={self.cfg.early_stopping_patience}, "
                    f"min_delta={self.cfg.early_stopping_delta}"
                )
                early_stopper_ae = EarlyStopping(
                    patience=self.cfg.early_stopping_patience,
                    min_delta=self.cfg.early_stopping_delta,
                )

            for epoch in range(1, self.cfg.epochs_pretrain + 1):
                E_full = self._encode_genes(X_gf, edge_in, edge_out, l_vec)
                rec_loss, dE = self._iterate_recon(X_cg, E_full)

                self.opt_enc.zero_grad()
                E_again = self._encode_genes(X_gf, edge_in, edge_out, l_vec)
                E_again.backward(dE)
                self.opt_enc.step()

                if epoch % self.cfg.verbose_every == 0 or epoch in (1, self.cfg.epochs_pretrain):
                    print(f"[Pretrain-AE ] epoch {epoch:4d}/{self.cfg.epochs_pretrain}  recon={rec_loss:.6f}")

                # --- 早停检查（以 recon 为准）---
                if early_stopper_ae is not None:
                    early_stopper_ae(rec_loss)

                    if early_stopper_ae.is_best:
                        save_current_model()

                    if early_stopper_ae.early_stop:
                        print(f"[Pretrain-AE ] Early stopping triggered at epoch {epoch}")
                        break

            # 预训练结束：若启用早停，则把权重回滚到最佳点
            if early_stopper_ae is not None:
                print("[Pretrain-AE ] Loading best model weights...")
                checkpoint = torch.load(model_save_path, map_location=self.device)
                self.encoder.load_state_dict(checkpoint["encoder"])
                self.decoder.load_state_dict(checkpoint["decoder"])
                self.dgi.load_state_dict(checkpoint["dgi"])
            else:
                # 没启用早停：保存最终一次（方便后续只有 pretrain 时也有 model.pt）
                save_current_model()

        # ===== C) 联合训练 (Joint) - 此处加入早停 =====
        if self.cfg.epochs_joint > 0:

            # 初始化早停
            early_stopper = None
            if self.cfg.early_stopping_patience > 0:
                print(f"[Joint] 启用早停机制: patience={self.cfg.early_stopping_patience}")
                early_stopper = EarlyStopping(
                    patience=self.cfg.early_stopping_patience,
                    min_delta=self.cfg.early_stopping_delta
                )

            for epoch in range(1, self.cfg.epochs_joint + 1):
                # AE 路径
                E_full = self._encode_genes(X_gf, edge_in, edge_out, l_vec)
                rec_loss, dE = self._iterate_recon(X_cg, E_full)

                self.opt_enc.zero_grad()
                E_again = self._encode_genes(X_gf, edge_in, edge_out, l_vec)
                E_again.backward(dE)
                self.opt_enc.step()

                # DGI 路径
                dgi_loss = self._step_dgi(X_gf, edge_in, edge_out, l_vec)

                # 总 Loss (用于监控)
                total_loss = rec_loss + dgi_loss


                if epoch % self.cfg.verbose_every == 0 or epoch in (1, self.cfg.epochs_joint):
                    print(
                        f"[Joint      ] epoch {epoch:4d}/{self.cfg.epochs_joint}  "
                        f"loss={total_loss:.4f} (rec={rec_loss:.4f}, dgi={dgi_loss:.4f})"
                    )

                # --- 早停检查 ---
                if early_stopper is not None:
                    early_stopper(total_loss)  # 这里我们监控训练集的总Loss（如果此时没有验证集）

                    if early_stopper.is_best:
                        # 保存当前最佳模型
                        save_current_model()
                        # print(f"  -> Best model saved at epoch {epoch}")

                    if early_stopper.early_stop:
                        print(f"[Joint] Early stopping triggered at epoch {epoch}")
                        break

            # 如果没有启用早停，或者训练结束时还没触发早停，最后保存一次
            if early_stopper is None:
                save_current_model()
            else:
                # 如果启用了早停，为了保证后续 return 的 E_final 是最佳模型的
                # 我们需要重新加载最佳权重
                print("[Joint] Loading best model weights for final inference...")
                checkpoint = torch.load(model_save_path, map_location=self.device)
                self.encoder.load_state_dict(checkpoint["encoder"])
                self.decoder.load_state_dict(checkpoint["decoder"])
                self.dgi.load_state_dict(checkpoint["dgi"])

        # ===== 导出结果 (使用当前(最佳)权重计算) =====
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()  # 确保 eval 模式

            E_final = (
                self._encode_genes(X_gf, edge_in, edge_out, l_vec)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            E_t = torch.from_numpy(E_final).to(self.device)
            X_all = torch.from_numpy(X_cg).to(self.device)
            Z_all, rowsum = self.projector(X_all, E_t)
            Z_cells = Z_all.cpu().numpy().astype(np.float32)

        np.save(os.path.join(out_dir, "Z_genes.npy"), E_final)
        np.save(os.path.join(out_dir, "Z_cells.npy"), Z_cells)

        # 保存 config json
        with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "out_dim": self.cfg.out_dim,
                    "netmodel_version": __version__,
                    "recon_space": self.cfg.recon_space,
                    "decoder_kind": self.cfg.decoder_kind,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return E_final, Z_cells

    # ===== 推理 / KO 接口（可选）=====
    @torch.no_grad()
    def decode_from_latent(
        self,
        Z_cells: np.ndarray,
        E_genes: np.ndarray,
        rowsum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        Z = torch.from_numpy(Z_cells).to(self.device)
        E = torch.from_numpy(E_genes).to(self.device)
        rs = torch.from_numpy(rowsum).to(self.device) if rowsum is not None else None
        Xhat = self.decoder.decode_from_latent(Z, E, rs)
        return Xhat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def simulate_gene_knockout(
        self,
        Z_cells: np.ndarray,
        E_genes: np.ndarray,
        ko_gene_idx: np.ndarray,
        strength: float = 1.0,
        rowsum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        简单 in-silico KO：对对称/GRN 解码器来说，通过衰减指定基因行的 embedding。
        对 MLPDecoder（与 E 无关）来说，KO 等价于原样解码。
        """
        E = torch.from_numpy(E_genes.copy()).to(self.device)
        if hasattr(self.decoder, "bias"):
            E[ko_gene_idx] *= (1.0 - strength)

        Z = torch.from_numpy(Z_cells).to(self.device)
        rs = torch.from_numpy(rowsum).to(self.device) if rowsum is not None else None
        Xhat = self.decoder.decode_from_latent(Z, E, rs)
        return Xhat.cpu().numpy().astype(np.float32)
