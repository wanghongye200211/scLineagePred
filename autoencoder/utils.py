# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# ---------- 预处理 ----------
def minmax_0_1(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x)
    xmin = np.min(x); xmax = np.max(x)
    denom = xmax - xmin
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (denom + eps)).astype(np.float32)

# 兼容旧写法
minmax01 = minmax_0_1
mimax01 = minmax_0_1  # 你之前用到的拼写也兼容

def log1p_if_needed(X: np.ndarray, do_log1p: bool = True) -> np.ndarray:
    X = np.asarray(X)
    if not do_log1p: return X.astype(np.float32)
    if (X < 0).any(): return X.astype(np.float32)
    return np.log1p(X).astype(np.float32)

# ---------- 先验网络 / 共表达 ----------
def read_prior_network(tsv_path: str, genes: np.ndarray):
    g2i = {g: i for i, g in enumerate(genes)}
    df = pd.read_csv(tsv_path, sep="\t", header=None, comment="#", dtype=str)
    if df.shape[1] < 2:
        raise ValueError("网络TSV需要至少两列（source, target）")
    src_names = df.iloc[:, 0].astype(str).values
    dst_names = df.iloc[:, 1].astype(str).values
    src, dst = [], []
    for s, d in zip(src_names, dst_names):
        if s in g2i and d in g2i:
            src.append(g2i[s]); dst.append(g2i[d])
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)

def build_coexp_knn(X_cg: np.ndarray, topk: int = 10):
    C, G = X_cg.shape
    X_g = X_cg.astype(np.float64).T
    X_g -= X_g.mean(axis=1, keepdims=True)
    std = X_g.std(axis=1, keepdims=True) + 1e-9
    Xn = X_g / std
    corr = (Xn @ Xn.T) / max(C - 1, 1)
    np.fill_diagonal(corr, -np.inf)
    src, dst = [], []
    for i in range(G):
        idx = np.argpartition(-corr[i], topk)[:topk]
        for j in idx:
            src.append(i); dst.append(int(j))
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)

def build_in_out_edges(src: np.ndarray, dst: np.ndarray):
    edges_in  = np.vstack([src, dst]).astype(np.int64)  # src -> dst
    edges_out = np.vstack([dst, src]).astype(np.int64)  # 反向，便于“out”通道
    return edges_in, edges_out


# utils.py (追加在末尾)

class EarlyStopping:
    """
    早停机制：当验证集损失（或训练损失）在 patience 个 epoch 内没有改善时停止训练。
    """
    def __init__(self, patience: int = 20, min_delta: float = 0.0, verbose: bool = False):
        """
        Args:
            patience (int): 损失不再下降后等待的 epoch 数。
            min_delta (float): 被认为是改善的最小变化量。
            verbose (bool): 是否打印日志。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.is_best = False # 标记当前 step 是否是最佳

    def __call__(self, val_loss: float):
        self.is_best = False
        if self.best_loss is None:
            self.best_loss = val_loss
            self.is_best = True
        elif val_loss > self.best_loss - self.min_delta:
            # 损失没有显著下降
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 损失下降了
            self.best_loss = val_loss
            self.counter = 0
            self.is_best = True