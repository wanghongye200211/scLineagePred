# -*- coding: utf-8 -*-
"""
Step4 (GSE132188) — v1-style pipeline
------------------------------------
目标：把你当前的 132188 Step4(v2) 改成 114412 Step4(v1) 的结构/输出。

输入：
- RUOT_INPUT_CSV   : preprocess_final/ruot_input_pca50_forward.csv   (samples + x1..x50)
- RUOT_MAPPING_TSV : preprocess_final/ruot_mapping_pca50_forward.tsv (row_id -> cell_id, samples, state, original_time...)
- EVAL_DIR         : DeepRUOT/results/GSE132188 (sde_point_*.npy 以及可选 sde_weight_*.npy)
- AE_H5AD          : processed/GSE132188_with_latent.h5ad (真实表达矩阵 + var_names + obsm['X_latent'])

输出（与 v1 一致）：
1) OUT_H5AD: processed/GSE132188_with_latent_and_clone.h5ad
   - X            : 真实表达矩阵（用于回归/按基因名切片）
   - obsm[X_latent] : autoencoder latent (例如 64D)
   - obs          : row_id/cell_id/samples/state/clone_root/clone_id 等

2) processed/pseudoclone_sequences.csv
3) processed/pseudoclone_seq_clone.npy
4) processed/pseudoclone_clone_map.tsv

说明（相对 v2 的关键变化）：
- 不再依赖 latent_h5ad 的 obs 来提供 label：优先使用 mapping.tsv 的 state（更标准、可追溯）
- 支持 sde_weight（如果没有则自动 fallback 为全 1）
- 生成 clone_id / seq_clone / clone_map
- 支持序列去重（DEDUP_BY_PATH）与 KNN batch（避免内存炸）

# --- Build direction ---
REVERSE_BUILD = True   # True: build sequences anchored at terminal time (reverse time order)


"""

import os
import glob
import re
import csv
import hashlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
import yaml

from DeepRUOT.models import FNet

# =============== 你只需要改这里（路径） ===============
RUOT_INPUT_CSV = "/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/ruot_input_pca50_forward.csv"
RUOT_MAPPING_TSV = "/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/ruot_mapping_pca50_forward.tsv"

EVAL_DIR = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/results/GSE132188"
OUT_DIR = "/Users/wanghongye/python/scLineagetracer/GSE132188/processed"

OUT_H5AD = "/Users/wanghongye/python/scLineagetracer/GSE132188/processed/GSE132188_with_latent_and_clone.h5ad"

# Step3 产物：必须包含真实表达矩阵 X + 基因名 var_names + obsm['X_latent']
AE_H5AD = "/Users/wanghongye/python/scLineagetracer/GSE132188/processed/GSE132188_with_latent.h5ad"

OUT_SEQ_CSV = os.path.join(OUT_DIR, "pseudoclone_sequences_reverse.csv")
OUT_SEQ_CLONE_NPY = os.path.join(OUT_DIR, "pseudoclone_seq_clone_reverse.npy")
OUT_CLONE_MAP_TSV = os.path.join(OUT_DIR, "pseudoclone_clone_map_reverse.tsv")
# =====================================================

# --- Build direction ---
# True: 以末端时间点为锚点，按反向时间构建序列
REVERSE_BUILD = True
# reverse 时锚点策略：
# - "particle": 粒子先匹配终点（旧逻辑）
# - "terminal_cell": 每个终点细胞直接作为锚点（推荐，保证 Delta 不会被粒子分布吞掉）
REVERSE_ANCHOR_MODE = "terminal_cell"
REVERSE_ANCHOR_PARTICLE_K = 8
REVERSE_ANCHOR_TAU = 1.0
WRITE_CLONE_METADATA = False

# 反向积分配置（从终点真实细胞出发）
MODEL_DIR = EVAL_DIR
MODEL_CONFIG_PATH = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/config/GSE132188.yaml"
INTEGRATION_DT = 0.05
INTEGRATION_BATCH = 512

LATENT_KEY = "X_latent"
TIME_COL = "samples"  # ruot_input 的时间列
STATE_COL = "state"  # mapping.tsv 中用于 endpoint_type 的列名（Step2 会把你配置的注释列写到这一列）

# 只保留能映射到 4 类内分泌的终点（与 Step5 的 LABEL_MAPPING 一致）
LABEL_MAPPING = {
    "Alpha": "Alpha", "Fev+ Alpha": "Alpha",
    "Beta": "Beta", "Fev+ Beta": "Beta",
    "Delta": "Delta", "Fev+ Delta": "Delta",
    "Epsilon": "Epsilon", "Fev+ Epsilon": "Epsilon",
}

# Step4 类别采样（默认关闭：四类都全保留）
USE_CLASS_SAMPLING = False
KEEP_PROB = {
    "Alpha": 1.00,
    "Beta": 1.00,
    "Delta": 1.00,
    "Epsilon": 1.00,
}

# KNN 参数（v1 可调）
KNN_K = 1
TAU = 2.0

# 每个 particle 生成几条序列（v1 默认 1；你原 v2 是 50）
SEQ_PER_PARTICLE = 50

SEED = 2026

# 跑哪些 run
USE_RUNS = None  # None=全部；例如 [0]

# 去重：完全相同 idx_t0..idx_tT-1 的序列只保留一次
DEDUP_BY_PATH = True

# KNN batch（大数据集时建议 10000~50000）
KNN_BATCH = 20000

# 是否写 OUT_H5AD
WRITE_OUT_H5AD = True
OVERWRITE_OUT_H5AD = True

_RUN_RE = re.compile(r"sde_point_(\d+)\.npy$")


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def sorted_x_cols(df: pd.DataFrame, prefix="x"):
    cols = [c for c in df.columns if c.startswith(prefix)]

    def key(c):
        tail = c[len(prefix):]
        return int(tail) if tail.isdigit() else 10 ** 9

    cols.sort(key=key)
    return cols


def load_sde_point(path: str) -> np.ndarray:
    Z = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if Z.ndim != 3:
        raise RuntimeError(f"sde_point must be 3D, got {Z.shape}")
    return Z


def load_sde_weight(path: str, T_true: int, N: int) -> np.ndarray:
    """
    DeepRUOT 有时会输出 sde_weight_{rid}.npy。
    - 若不存在：fallback 为全 1
    - 若形状为 (T,N) -> 扩维到 (T,N,1)
    - 若转置为 (N,T,1) -> 自动转回 (T,N,1)
    """
    if (path is None) or (not os.path.isfile(path)):
        W = np.ones((T_true, N, 1), dtype=np.float32)
        return W

    w = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if w.ndim == 2:
        w = w[:, :, None]
    if w.ndim != 3:
        raise RuntimeError(f"sde_weight must be 3D, got {w.shape}")

    # normalize to (T,N,1)
    if w.shape[0] != T_true and w.shape[1] == T_true:
        w = np.transpose(w, (1, 0, 2))
    if w.shape[0] != T_true or w.shape[1] != N:
        # 仍不匹配就 fallback（避免你因 weight 输出不一致而跑不起来）
        W = np.ones((T_true, N, 1), dtype=np.float32)
        return W
    return w


def soft_prob_from_dist(distK: np.ndarray, tau: float) -> np.ndarray:
    if distK.ndim == 2 and distK.shape[1] == 1:
        return np.ones((distK.shape[0], 1), dtype=np.float32)
    d2 = (distK.astype(np.float64) ** 2)
    P = np.exp(-d2 / (float(tau) ** 2 + 1e-12))
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return P.astype(np.float32)


def choose_with_prob(rng: np.random.Generator, cands: np.ndarray, p: np.ndarray) -> int:
    """
    稳健抽样：
    - 自动重归一化
    - 出现 NaN/全0/负数时回退到均匀抽样
    """
    c = np.asarray(cands, dtype=np.int64)
    if c.size == 0:
        raise ValueError("empty candidates")
    if c.size == 1:
        return int(c[0])

    pp = np.asarray(p, dtype=np.float64).reshape(-1)
    if pp.shape[0] != c.shape[0]:
        return int(rng.choice(c))
    pp[~np.isfinite(pp)] = 0.0
    pp = np.maximum(pp, 0.0)
    s = float(pp.sum())
    if s <= 0.0:
        return int(rng.choice(c))
    pp /= s
    return int(rng.choice(c, p=pp))


def pick_time_label_col(mp: pd.DataFrame) -> str:
    for c in ("diffday", "original_time", "samples"):
        if c in mp.columns:
            return c
    return ""


def pick_clone_col(mp: pd.DataFrame) -> str:
    for c in ["clone_root", "clone", "cloneID", "clone_id", "clonal_id", "clone_barcode", "barcode"]:
        if c in mp.columns:
            return c
    return ""


def path_hash(path_rows: np.ndarray) -> int:
    b = path_rows.astype(np.int32, copy=False).tobytes()
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def kneighbors_batched(nn: NearestNeighbors, Xq: np.ndarray, k: int, batch: int):
    N = Xq.shape[0]
    dist_all = np.empty((N, k), dtype=np.float32)
    ind_all = np.empty((N, k), dtype=np.int64)
    for st in range(0, N, batch):
        ed = min(st + batch, N)
        dist, ind = nn.kneighbors(Xq[st:ed], n_neighbors=k, return_distance=True)
        dist_all[st:ed] = dist.astype(np.float32)
        ind_all[st:ed] = ind.astype(np.int64)
    return dist_all, ind_all


def list_runs(eval_dir: str):
    runs = []
    for p in sorted(glob.glob(os.path.join(eval_dir, "sde_point_*.npy"))):
        m = _RUN_RE.search(p)
        if not m:
            continue
        rid = int(m.group(1))
        runs.append(rid)
    return sorted(set(runs))


def resolve_device_from_config(cfg: dict):
    dev = str(cfg.get("device", "cpu")).lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_training_config():
    if os.path.isfile(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    params_path = os.path.join(MODEL_DIR, "params.yml")
    if os.path.isfile(params_path):
        with open(params_path, "r") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Missing config: {MODEL_CONFIG_PATH} and {params_path}")


def load_velocity_model():
    cfg = load_training_config()
    mc = cfg.get("model", {})
    need_keys = ["in_out_dim", "hidden_dim", "n_hiddens", "activation"]
    miss = [k for k in need_keys if k not in mc]
    if miss:
        raise KeyError(f"Missing model keys in config: {miss}")

    device = resolve_device_from_config(cfg)
    f_net = FNet(
        in_out_dim=int(mc["in_out_dim"]),
        hidden_dim=int(mc["hidden_dim"]),
        n_hiddens=int(mc["n_hiddens"]),
        activation=str(mc["activation"]),
    ).to(device)

    model_path = os.path.join(MODEL_DIR, "model_final")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_final not found: {model_path}")
    f_net.load_state_dict(torch.load(model_path, map_location=device))
    f_net.eval()
    for p in f_net.parameters():
        p.requires_grad_(False)
    return f_net, device


@torch.no_grad()
def integrate_reverse_velocity(f_net, x_terminal: np.ndarray, times_desc: list, dt: float, batch_size: int, device):
    """
    从终点细胞 x_terminal 出发，按速度场 v(t,x) 反向积分，输出 (T, N, D)。
    times_desc 必须是降序，例如 [3,2,1,0]。
    """
    if len(times_desc) < 2:
        raise ValueError("Need at least two time points")
    if any(times_desc[i] < times_desc[i + 1] for i in range(len(times_desc) - 1)):
        raise ValueError(f"times_desc must be descending, got {times_desc}")

    N, D = x_terminal.shape
    T = len(times_desc)
    dt_abs = float(abs(dt))
    if dt_abs <= 0:
        raise ValueError(f"INTEGRATION_DT must be > 0, got {dt}")

    traj = np.empty((T, N, D), dtype=np.float32)
    traj[0] = x_terminal.astype(np.float32, copy=False)

    x_cur = x_terminal.astype(np.float32, copy=True)
    for ti in range(1, T):
        t_start = float(times_desc[ti - 1])
        t_end = float(times_desc[ti])
        if t_end > t_start:
            raise ValueError(f"times_desc[{ti}] should be <= previous, got {times_desc}")

        for st in range(0, N, batch_size):
            ed = min(st + batch_size, N)
            xb = torch.from_numpy(x_cur[st:ed]).to(device)
            t_now = t_start
            while t_now > t_end + 1e-8:
                h = min(dt_abs, t_now - t_end)
                t_tensor = torch.tensor([t_now], dtype=torch.float32, device=device)
                v = f_net.v_net(t_tensor, xb)
                xb = xb - v * h
                t_now -= h
            x_cur[st:ed] = xb.detach().cpu().numpy().astype(np.float32)
        traj[ti] = x_cur
    return traj


def maybe_write_out_h5ad(df_input: pd.DataFrame, mp: pd.DataFrame):
    """
    写 OUT_H5AD，关键约束：
    1) adata.X 必须是真实表达矩阵（用于回归/按基因名切片）
    2) adata.obsm[LATENT_KEY] 必须是 autoencoder latent
    3) obs 里补齐 row_id/cell_id/samples/state/clone_root/clone_id 等字段
    """
    if not WRITE_OUT_H5AD:
        return
    if os.path.isfile(OUT_H5AD) and (not OVERWRITE_OUT_H5AD):
        print(f"[H5AD] exists -> skip: {OUT_H5AD}")
        return

    import scanpy as sc

    N = len(mp)
    if len(df_input) != N:
        raise RuntimeError(f"[H5AD] row mismatch: ruot_input={len(df_input)} mapping={N}")

    mp = mp.sort_values("row_id").reset_index(drop=True)
    row_id = mp["row_id"].to_numpy(np.int64)
    if not (row_id[0] == 0 and row_id[-1] == N - 1 and np.all(row_id == np.arange(N))):
        raise RuntimeError("[H5AD] mapping.row_id must be 0..N-1 and sorted")

    if not os.path.isfile(AE_H5AD):
        raise FileNotFoundError(
            f"[H5AD] AE_H5AD not found: {AE_H5AD}\n"
            f"请先跑 Step3 生成 GSE132188_with_latent.h5ad"
        )

    print(f"[H5AD] loading AE h5ad: {AE_H5AD}")
    adata_ae = sc.read_h5ad(AE_H5AD)

    if LATENT_KEY not in adata_ae.obsm:
        raise KeyError(f"[H5AD] AE h5ad missing obsm/{LATENT_KEY}, got {list(adata_ae.obsm.keys())}")

    map_cell_id = mp["cell_id"].astype(str).to_numpy()
    if "cell_id" in adata_ae.obs.columns:
        ae_cell_id = adata_ae.obs["cell_id"].astype(str).to_numpy()
    else:
        ae_cell_id = adata_ae.obs_names.astype(str).to_numpy()

    ae_index = pd.Index(ae_cell_id)
    order = ae_index.get_indexer(map_cell_id)
    missing = np.where(order < 0)[0]
    if len(missing) > 0:
        ex = missing[:10]
        raise RuntimeError(
            f"[H5AD] {len(missing)} mapping cell_id not found in AE h5ad.\n"
            f"  examples row_id={ex.tolist()}, cell_id={map_cell_id[ex].tolist()}\n"
            f"  解决：确保 Step2/Step3 用的 cell_id 与 mapping.tsv 的 cell_id 同一套。"
        )

    adata = adata_ae[order].copy()

    # 覆盖/补齐 obs 字段（以 mapping 为准）
    adata.obs["row_id"] = row_id
    adata.obs["cell_id"] = map_cell_id
    adata.obs["samples"] = mp["samples"].to_numpy(np.int32)

    if STATE_COL in mp.columns:
        adata.obs["state"] = mp[STATE_COL].astype(str).to_numpy()
    else:
        # 如果 Step2 没写 state，这里给个占位，至少不崩
        adata.obs["state"] = np.array([""] * N, dtype=object)

    # 额外字段（如果 mapping 有就带上）
    if "diffday" in mp.columns:
        adata.obs["diffday"] = mp["diffday"].astype(str).to_numpy()
    if "original_time" in mp.columns:
        adata.obs["original_time"] = mp["original_time"].astype(str).to_numpy()
    if "parsed_time" in mp.columns:
        adata.obs["parsed_time"] = mp["parsed_time"].to_numpy()

    # clone_root/clone_id（mapping 没有 clone 字段时，退化为每个 cell 自己一个 clone）
    clone_col = pick_clone_col(mp)
    if clone_col:
        clone_root = mp[clone_col].astype(str).to_numpy()
    else:
        clone_root = map_cell_id

    clone_id, _ = pd.factorize(clone_root, sort=True)
    adata.obs["clone_root"] = clone_root
    adata.obs["clone_id"] = clone_id.astype(np.int64)

    adata.uns["latent_key"] = LATENT_KEY

    os.makedirs(os.path.dirname(OUT_H5AD), exist_ok=True)
    print(f"[H5AD] writing -> {OUT_H5AD}")
    adata.write_h5ad(OUT_H5AD, compression="gzip")
    print("[H5AD] done.")


def map_endpoint_label(raw: str):
    raw = str(raw).strip()
    if raw in LABEL_MAPPING:
        return LABEL_MAPPING[raw]
    # 兜底：用关键词匹配（防止标签里带额外前后缀）
    for k in ("Alpha", "Beta", "Delta", "Epsilon"):
        if k in raw:
            return k
    return None


def main():
    np.random.seed(SEED)
    ensure_out_dir()
    deterministic_top1 = (int(KNN_K) == 1)
    seq_per_particle_eff = 1 if deterministic_top1 else int(SEQ_PER_PARTICLE)
    print(
        f"[CONFIG] REVERSE_BUILD={REVERSE_BUILD} "
        f"KNN_K={KNN_K} mode={'top1' if deterministic_top1 else f'top{KNN_K}_stochastic'} "
        f"SEQ_PER_PARTICLE={SEQ_PER_PARTICLE} effective={seq_per_particle_eff} "
        f"DEDUP_BY_PATH={DEDUP_BY_PATH}"
    )

    # 防御：csv 如果被建成目录会直接炸
    if os.path.isdir(OUT_SEQ_CSV):
        raise IsADirectoryError(
            f"{OUT_SEQ_CSV} 是目录，请先删掉：\nrm -rf {OUT_SEQ_CSV}\n"
        )

    # ---- load ruot_input ----
    print(f"[INFO] ruot_input: {RUOT_INPUT_CSV}")
    df = pd.read_csv(RUOT_INPUT_CSV)
    if TIME_COL not in df.columns:
        raise KeyError(f"Missing TIME_COL='{TIME_COL}' in ruot_input")

    x_cols = sorted_x_cols(df, "x")
    if not x_cols:
        raise RuntimeError("No x1..xK columns in ruot_input")

    X_real = df[x_cols].to_numpy(np.float32)
    times = df[TIME_COL].to_numpy(np.int32)
    uniq_times = sorted(np.unique(times).tolist())
    T_true = len(uniq_times)
    rows_by_time = {t: np.where(times == t)[0] for t in uniq_times}
    print(f"[INFO] T={T_true} uniq_times={uniq_times} N_total={len(df)} D={X_real.shape[1]}")

    # ---- load mapping ----
    print(f"[INFO] mapping: {RUOT_MAPPING_TSV}")
    mp = pd.read_csv(RUOT_MAPPING_TSV, sep="\t").sort_values("row_id").reset_index(drop=True)
    required = {"row_id", "cell_id", "samples"}
    miss = required - set(mp.columns)
    if miss:
        raise KeyError(f"Mapping missing columns: {miss}. Got={list(mp.columns)}")

    if len(mp) != len(df):
        raise RuntimeError(f"Row mismatch: mapping={len(mp)} vs ruot_input={len(df)}")

    row_id = mp["row_id"].to_numpy(np.int64)
    if not (row_id[0] == 0 and row_id[-1] == len(df) - 1 and np.all(row_id == np.arange(len(df)))):
        raise RuntimeError("Mapping row_id not contiguous 0..N-1")

    # ---- 先写 OUT_H5AD（v1 的标准产物）----
    maybe_write_out_h5ad(df, mp)

    # ---- label source ----
    if STATE_COL in mp.columns:
        state = mp[STATE_COL].astype(str).to_numpy()
        print(f"[INFO] endpoint label source: mapping['{STATE_COL}']")
    else:
        # 兜底：从 AE_H5AD 读指定列（不推荐，但至少能跑）
        import scanpy as sc
        ad = sc.read_h5ad(AE_H5AD)
        if "clusters_fig6_fine_final" in ad.obs.columns:
            state = ad.obs["clusters_fig6_fine_final"].astype(str).to_numpy()
            print("[WARN] mapping has no 'state'; fallback to AE_H5AD.obs['clusters_fig6_fine_final']")
        else:
            raise KeyError(f"Mapping has no '{STATE_COL}', and AE_H5AD has no fallback label column.")

    cell_id = mp["cell_id"].astype(str).to_numpy()

    time_label_col = pick_time_label_col(mp)
    time_label = mp[time_label_col].astype(str).to_numpy() if time_label_col else None
    print(f"[INFO] time label col = {time_label_col if time_label_col else '(none)'}")

    # ---- knn per time ----
    nn_real = {}
    for t in uniq_times:
        idx = rows_by_time[t]
        nn = NearestNeighbors(n_neighbors=min(KNN_K, len(idx)), metric="euclidean")
        nn.fit(X_real[idx])
        nn_real[t] = (nn, idx)

    if not REVERSE_BUILD:
        raise NotImplementedError("This script currently supports REVERSE_BUILD=True only.")

    # ---- runs ----
    runs = list_runs(EVAL_DIR)
    if USE_RUNS is not None:
        runs = [r for r in runs if r in set(USE_RUNS)]
    if not runs:
        runs = [0]
    if deterministic_top1 and USE_RUNS is None and len(runs) > 1:
        runs = [runs[0]]
        print(f"[INFO] KNN_K=1 detected; collapse runs to single run: {runs}")
    print(f"[INFO] runs = {runs}")

    # ---- reverse integration setup ----
    f_net, model_device = load_velocity_model()
    times_build = list(uniq_times)[::-1]  # e.g. [3,2,1,0]
    t_terminal = times_build[0]

    # 终点细胞（带四类标签）
    terminal_rows_all = rows_by_time[t_terminal]
    terminal_rows = np.array(
        [r for r in terminal_rows_all if map_endpoint_label(state[int(r)]) is not None],
        dtype=np.int64,
    )
    if terminal_rows.shape[0] == 0:
        raise RuntimeError(f"No mapped terminal cells found at t={t_terminal}")

    term_clean = [map_endpoint_label(state[int(r)]) for r in terminal_rows]
    term_stat = pd.Series(term_clean).value_counts().to_dict()
    print(
        f"[INFO] reverse-integrate from terminal time t={t_terminal}, "
        f"n_terminal={len(terminal_rows)}, label_counts={term_stat}, device={model_device}"
    )

    # ---- CSV outputs ----
    header = ["seq_id", "direction", "run_id", "particle_id", "rep_id"]
    if WRITE_CLONE_METADATA:
        header += ["clone_root", "clone_id"]
    header += [
        "endpoint", "endpoint_type", "endpoint_time",
        "endpoint_clean", "origin", "origin_type", "origin_time", "samples_order", "keep_prob",
        "w_end"
    ]
    header += [f"w_t{i}" for i in range(T_true)]
    header += [f"idx_t{i}" for i in range(T_true)]
    header += [f"id_t{i}" for i in range(T_true)]

    clone_root_to_id = {}
    clone_id_to_root = []
    clone_counts = []

    seen = set() if DEDUP_BY_PATH else None

    seq_clone = []
    kept_clean = {k: 0 for k in KEEP_PROB.keys()}
    total_written = 0
    total_candidates = 0

    os.makedirs(os.path.dirname(OUT_SEQ_CSV), exist_ok=True)

    print(f"[CSV] writing: {OUT_SEQ_CSV}")
    with open(OUT_SEQ_CSV, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        writer.writeheader()

        for rid in runs:
            run_seed = SEED + int(rid)
            rng = np.random.default_rng(run_seed)

            # 1) 以终点真实细胞 PCA 为初值反向积分
            x_terminal = X_real[terminal_rows]
            Zb = integrate_reverse_velocity(
                f_net=f_net,
                x_terminal=x_terminal,
                times_desc=times_build,
                dt=INTEGRATION_DT,
                batch_size=INTEGRATION_BATCH,
                device=model_device,
            )  # (T,N,D)
            N = Zb.shape[1]

            # 2) 各时间点积分结果 -> 真实细胞 KNN 候选
            cand_idx = [None] * T_true
            cand_prob = [None] * T_true
            for ti, t in enumerate(times_build):
                nn_t, rows_t = nn_real[t]
                k = min(KNN_K, len(rows_t))
                distK, indK = kneighbors_batched(nn_t, Zb[ti], k, KNN_BATCH)
                cand_idx[ti] = rows_t[indK]
                cand_prob[ti] = None if deterministic_top1 else soft_prob_from_dist(distK, TAU)

            # 3) 序列生成：终点行号固定为真实终点细胞
            kept_run = {k: 0 for k in KEEP_PROB.keys()}
            for i, term_row in enumerate(terminal_rows):
                term_row = int(term_row)
                term_type = str(state[term_row])
                clean = map_endpoint_label(term_type)
                if clean is None:
                    continue

                keep_p = float(KEEP_PROB.get(clean, 1.0)) if USE_CLASS_SAMPLING else 1.0
                w_t = np.ones((T_true,), dtype=np.float32)

                for rep in range(seq_per_particle_eff):
                    total_candidates += 1
                    if USE_CLASS_SAMPLING and (rng.random() > keep_p):
                        continue

                    path_rows = np.empty((T_true,), dtype=np.int64)
                    path_rows[0] = term_row
                    for ti in range(1, T_true):
                        cands = cand_idx[ti][i]
                        if deterministic_top1:
                            path_rows[ti] = int(cands[0])
                        else:
                            p = cand_prob[ti][i]
                            path_rows[ti] = choose_with_prob(rng, cands, p)

                    origin_row = int(path_rows[-1])

                    if DEDUP_BY_PATH:
                        h = path_hash(path_rows)
                        if h in seen:
                            continue
                        seen.add(h)

                    row = {
                        "seq_id": total_written,
                        "direction": "reverse",
                        "run_id": rid,
                        "particle_id": i,
                        "rep_id": rep,
                        "endpoint": str(cell_id[term_row]),
                        "endpoint_type": term_type,
                        "endpoint_time": (str(time_label[term_row]) if time_label is not None else ""),
                        "endpoint_clean": clean,
                        "origin": str(cell_id[origin_row]),
                        "origin_type": str(state[origin_row]),
                        "origin_time": (str(time_label[origin_row]) if time_label is not None else ""),
                        "samples_order": ",".join([str(x) for x in times_build]),
                        "keep_prob": keep_p,
                        "w_end": float(w_t[0]),
                    }
                    if WRITE_CLONE_METADATA:
                        clone_root = str(cell_id[term_row])
                        if clone_root in clone_root_to_id:
                            cid = clone_root_to_id[clone_root]
                            clone_counts[cid] += 1
                        else:
                            cid = len(clone_id_to_root)
                            clone_root_to_id[clone_root] = cid
                            clone_id_to_root.append(clone_root)
                            clone_counts.append(1)
                        row["clone_root"] = clone_root
                        row["clone_id"] = cid
                    for ti in range(T_true):
                        row[f"w_t{ti}"] = float(w_t[ti])
                        row[f"idx_t{ti}"] = int(path_rows[ti])
                        row[f"id_t{ti}"] = str(cell_id[int(path_rows[ti])])

                    writer.writerow(row)
                    if WRITE_CLONE_METADATA:
                        seq_clone.append(cid)

                    total_written += 1
                    kept_clean[clean] += 1
                    kept_run[clean] += 1

            print(
                f"[RUN {rid}] mode=terminal_cell+reverse_integrate "
                f"knn_mode={'top1' if deterministic_top1 else f'top{KNN_K}_stochastic'} "
                f"kept={kept_run} cumulative={kept_clean} written={total_written}"
            )

    # clone outputs（可选）
    if WRITE_CLONE_METADATA:
        np.save(OUT_SEQ_CLONE_NPY, np.array(seq_clone, dtype=np.int64))
        df_clone = pd.DataFrame({
            "clone_id": np.arange(len(clone_id_to_root), dtype=np.int64),
            "clone_root": np.array(clone_id_to_root, dtype=object),
            "n_sequences": np.array(clone_counts, dtype=np.int64),
        }).sort_values("n_sequences", ascending=False).reset_index(drop=True)
        df_clone.to_csv(OUT_CLONE_MAP_TSV, sep="\t", index=False)

    print("\n[DONE Step4]")
    print(f"  OUT_H5AD         : {OUT_H5AD}")
    print(f"  sequences CSV    : {OUT_SEQ_CSV} (written={total_written}, candidates={total_candidates})")
    if WRITE_CLONE_METADATA:
        print(f"  seq_clone npy    : {OUT_SEQ_CLONE_NPY}")
        print(f"  clone_map tsv    : {OUT_CLONE_MAP_TSV}")


if __name__ == "__main__":
    main()
