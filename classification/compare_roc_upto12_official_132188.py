# -*- coding: utf-8 -*-
"""
GSE132188 official-method comparison at UpTo_12.5.

Compared methods:
- scLineagetracer (saved models + stacking pkl, same test split)
- CellRank (official API)
- WOT (official API)
- CoSpar (official API, optional; slower)

Task:
- 4-class prediction: Alpha/Beta/Delta/Epsilon
- Evaluate on same sequence test split as scLineagetracer.
"""

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc, accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

# runtime env for scanpy/cellrank/cospar on this machine
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import scanpy as sc
import cellrank as cr
import wot
import cospar as cs


GREY = "#444444"
LINE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300


TARGET_CLASSES = ["Alpha", "Beta", "Delta", "Epsilon"]
STATE_CANON_MAP = {
    "Alpha": "Alpha",
    "Fev+ Alpha": "Alpha",
    "Beta": "Beta",
    "Fev+ Beta": "Beta",
    "Delta": "Delta",
    "Fev+ Delta": "Delta",
    "Epsilon": "Epsilon",
    "Fev+ Epsilon": "Epsilon",
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_fig(fig, out_no_ext):
    fig.savefig(out_no_ext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_no_ext + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _style_axes_clean(ax):
    ax.tick_params(axis="both", direction="out", top=False, right=False)
    ax.tick_params(labelbottom=False, labelleft=False)


def _roc_endpoints_clean(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    if (len(fpr) == 0) or (fpr[0] != 0.0) or (tpr[0] != 0.0):
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    if (fpr[-1] != 1.0) or (tpr[-1] != 1.0):
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)
    m1 = fpr == 1.0
    if np.any(m1):
        t1 = float(np.max(tpr[m1]))
        keep = ~m1
        fpr2 = np.append(fpr[keep], 1.0)
        tpr2 = np.append(tpr[keep], t1)
        o2 = np.argsort(fpr2)
        fpr, tpr = fpr2[o2], tpr2[o2]
    return fpr, tpr


def compute_macro_curve_auc(y_true, prob):
    c = prob.shape[1]
    y_bin = label_binarize(y_true, classes=np.arange(c))
    grid = np.linspace(0.0, 1.0, 201)
    tprs = []
    aucs = []
    for i in range(c):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        fpr, tpr = _roc_endpoints_clean(fpr, tpr)
        aucs.append(float(auc(fpr, tpr)))
        tpr_i = np.interp(grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(auc(grid, mean_tpr))
    return grid, mean_tpr, macro_auc, aucs


def infer_reverse_from_index_csv(index_csv_path):
    if (index_csv_path is None) or (not os.path.isfile(index_csv_path)):
        return False
    try:
        df = pd.read_csv(index_csv_path, usecols=["samples_order"])
        ser = df["samples_order"].dropna().astype(str)
        for raw in ser.head(100):
            toks = [t.strip() for t in raw.split(",") if t.strip() != ""]
            vals = []
            ok = True
            for t in toks:
                if t.lstrip("-").isdigit():
                    vals.append(int(t))
                else:
                    ok = False
                    break
            if ok and len(vals) >= 2:
                return vals[0] > vals[-1]
    except Exception:
        return False
    return False


def eval_multiclass(y_true, prob):
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    p = p / p.sum(axis=1, keepdims=True)
    pred = np.argmax(p, axis=1)
    grid, macro_tpr, macro_auc, aucs = compute_macro_curve_auc(y_true, p)
    return {
        "AUC_macro": macro_auc,
        "Accuracy": float(accuracy_score(y_true, pred)),
        "LogLoss": float(log_loss(y_true, p, labels=list(range(p.shape[1])))),
        "fpr_macro": grid,
        "tpr_macro": macro_tpr,
        "AUC_per_class": aucs,
    }


def plot_dual(curves, out_dir):
    base = os.path.join(out_dir, "ROC_Comparison_UpTo_12.5_Official")
    order = ["scLineagetracer", "CellRank", "WOT", "CoSpar"]
    color = {m: LINE_COLORS[i] for i, m in enumerate(order)}

    def draw(ax, clean):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=GREY, alpha=0.55)
        for m in order:
            if m not in curves:
                continue
            c = curves[m]
            ax.plot(c["fpr_macro"], c["tpr_macro"], lw=2.7, color=color[m], label=f"{m} (Macro AUC={c['AUC_macro']:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0.02, y=0.02)
        if clean:
            _style_axes_clean(ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        else:
            ax.tick_params(axis="both", direction="out", top=False, right=False)
            ax.set_xlabel("False Positive Rate", fontsize=13)
            ax.set_ylabel("True Positive Rate", fontsize=13)
            ax.set_title("Macro ROC (UpTo_12.5 Official)", fontsize=13, pad=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(False)

    fig1, ax1 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax1, clean=False)
    fig1.tight_layout()
    save_fig(fig1, base + "_full")

    fig2, ax2 = plt.subplots(figsize=(7.2, 6.3))
    draw(ax2, clean=True)
    fig2.tight_layout()
    save_fig(fig2, base + "_clean")
    return base


class SeqDataset(Dataset):
    def __init__(self, X, y, idx, keep_len):
        self.X = X
        self.y = y
        self.idx = np.asarray(idx, dtype=np.int64)
        self.keep_len = int(keep_len)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j][: self.keep_len]
        return torch.from_numpy(x), torch.tensor(int(self.y[j]), dtype=torch.long), torch.tensor(self.keep_len, dtype=torch.long)


def collate_pad(batch):
    xs, ys, lens = zip(*batch)
    lens = torch.stack(lens, dim=0)
    ys = torch.stack(ys, dim=0)
    max_len = int(lens.max().item())
    d = xs[0].shape[1]
    Xp = torch.zeros((len(xs), max_len, d), dtype=torch.float32)
    for i, x in enumerate(xs):
        L = x.shape[0]
        Xp[i, :L] = x
    return Xp, ys, lens


class BiLSTMModel(nn.Module):
    def __init__(self, d, h, n_layers, dropout, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(d, h, n_layers, batch_first=True, bidirectional=True, dropout=(dropout if n_layers > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h * 2), nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_classes))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(feat)


class RNNModel(nn.Module):
    def __init__(self, d, h, n_layers, dropout, n_classes):
        super().__init__()
        self.rnn = nn.RNN(d, h, n_layers, batch_first=True, dropout=(dropout if n_layers > 1 else 0.0))
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, n_classes))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(h[-1])


class TransformerBlock(nn.Module):
    def __init__(self, d, nhead, ff_mult=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * ff_mult, d), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, d, ff, n_layers, dropout, nhead, n_classes):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d, nhead, ff_mult=2, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, n_classes))

    def forward(self, x, lengths):
        b, t, _ = x.shape
        ar = torch.arange(t, device=x.device)[None, :].expand(b, t)
        key_padding_mask = ar >= lengths[:, None]
        h = x
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        mask = (~key_padding_mask).float().unsqueeze(-1)
        feat = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(feat)


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    probs = []
    for x, _, lengths in loader:
        logits = model(x.to(device), lengths.to(device))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def load_sclineage_upto12_probs(seq_h5, index_csv, model_dir):
    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)
        classes = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["classes"][:]]
        time_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["time_labels"][:]]
        te_idx = np.asarray(f["test_idx"], dtype=np.int64)

    if infer_reverse_from_index_csv(index_csv):
        X = X[:, ::-1, :].copy()

    keep_len = time_labels.index("12.5") + 1
    ds = SeqDataset(X, y, te_idx, keep_len=keep_len)
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_pad)

    device = torch.device("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    n_classes = len(classes)
    dim = X.shape[2]

    bilstm = BiLSTMModel(dim, 256, 2, 0.10, n_classes).to(device)
    rnn = RNNModel(dim, 256, 2, 0.10, n_classes).to(device)
    trans = TransformerModel(dim, 256, 2, 0.10, 4, n_classes).to(device)

    bilstm.load_state_dict(torch.load(os.path.join(model_dir, "UpTo_12.5_BiLSTM_s2026.pth"), map_location=device))
    rnn.load_state_dict(torch.load(os.path.join(model_dir, "UpTo_12.5_RNN_s2026.pth"), map_location=device))
    trans.load_state_dict(torch.load(os.path.join(model_dir, "UpTo_12.5_Trans_s2026.pth"), map_location=device))

    p1 = get_probs(bilstm, dl, device)
    p2 = get_probs(rnn, dl, device)
    p3 = get_probs(trans, dl, device)

    with open(os.path.join(model_dir, "UpTo_12.5_Stacking_s2026.pkl"), "rb") as f:
        stacker = pickle.load(f)
    p_stack = stacker.predict_proba(np.concatenate([p1, p2, p3], axis=1))
    y_true = y[te_idx]
    return te_idx, y_true, p_stack, classes


def prepare_subset(adata, source_ids, time_col, state_col_model, source_time=12.5, target_time=15.5, mid_times=(13.5, 14.5), mid_cap_each=3000):
    time = adata.obs[time_col].astype(float).to_numpy()
    state = adata.obs[state_col_model].astype(str).to_numpy()

    source_set = set(str(x) for x in source_ids)
    source_mask = np.array([str(x) in source_set for x in adata.obs_names], dtype=bool)
    target_mask = (time == float(target_time)) & np.isin(state, TARGET_CLASSES)

    mid_mask = np.zeros(adata.n_obs, dtype=bool)
    rng = np.random.default_rng(2026)
    for mt in mid_times:
        idx = np.where(time == float(mt))[0]
        if len(idx) > mid_cap_each:
            idx = rng.choice(idx, size=mid_cap_each, replace=False)
        mid_mask[idx] = True

    keep = source_mask | target_mask | mid_mask
    sub = adata[keep].copy()
    sub.obs_names = pd.Index([str(x) for x in sub.obs_names])
    return sub


def extract_source_matrix_by_ids(prob_by_cell_mat, source_ids):
    out = []
    for cid in source_ids:
        row = prob_by_cell_mat.get(str(cid))
        if row is None:
            raise RuntimeError(f"Missing probabilities for source cell {cid}.")
        out.append(row)
    return np.vstack(out)


def run_cellrank_multiclass(adata_sub, time_col, state_col_model, source_time, target_time):
    work = adata_sub.copy()
    sc.pp.pca(work, n_comps=min(50, work.n_obs - 1, work.n_vars - 1))
    sc.pp.neighbors(work, n_neighbors=min(30, work.n_obs - 1), use_rep="X_pca")
    ck = cr.kernels.ConnectivityKernel(work).compute_transition_matrix()
    pk = cr.kernels.PseudotimeKernel(work, time_key=time_col).compute_transition_matrix(
        threshold_scheme="hard",
        n_jobs=1,
        backend="threading",
        show_progress_bar=False,
    )
    kernel = 0.5 * ck + 0.5 * pk
    g = cr.estimators.GPCCA(kernel)

    time = work.obs[time_col].astype(float).to_numpy()
    state = work.obs[state_col_model].astype(str).to_numpy()
    term = {}
    for cls in TARGET_CLASSES:
        term[cls] = list(work.obs_names[(time == float(target_time)) & (state == cls)])
    g.set_terminal_states(term)
    g.compute_fate_probabilities(
        solver="direct",
        use_petsc=False,
        n_jobs=1,
        backend="threading",
        show_progress_bar=False,
    )

    fp = g.fate_probabilities
    names = list(fp.names)
    mat = np.asarray(fp.X, dtype=np.float64)
    col_idx = [names.index(c) for c in TARGET_CLASSES]
    mat = mat[:, col_idx]
    cell_ids = np.array(work.obs_names, dtype=str)
    return {str(cid): mat[i] for i, cid in enumerate(cell_ids)}


def run_wot_multiclass(adata_sub, time_col, state_col_model, source_time, target_time):
    work = adata_sub.copy()
    work.obs["day_wot"] = work.obs[time_col].astype(float)
    model = wot.ot.OTModel(work, day_field="day_wot")
    tm = model.compute_transport_map(float(source_time), float(target_time))
    M = np.asarray(tm.X, dtype=np.float64)
    target_labels = work.obs.loc[tm.var_names, state_col_model].astype(str).to_numpy()

    probs = np.zeros((M.shape[0], len(TARGET_CLASSES)), dtype=np.float64)
    for j, cls in enumerate(TARGET_CLASSES):
        mask = target_labels == cls
        probs[:, j] = M[:, mask].sum(axis=1)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    source_ids = np.array(tm.obs_names, dtype=str)
    return {str(cid): probs[i] for i, cid in enumerate(source_ids)}


def run_cospar_multiclass(adata_sub, time_col, state_col_model, source_time, target_time, out_dir):
    work = adata_sub.copy()
    work.obs[time_col] = work.obs[time_col].astype(str)
    sc.pp.pca(work, n_comps=min(30, work.n_obs - 1, work.n_vars - 1))

    cs.settings.data_path = out_dir
    cs.settings.figure_path = out_dir
    cs.settings.verbosity = 1

    ac = cs.pp.initialize_adata_object(
        adata=None,
        X_state=work.X,
        X_pca=work.obsm["X_pca"],
        cell_names=np.array(work.obs_names),
        gene_names=np.array(work.var_names),
        time_info=work.obs[time_col].to_numpy(),
        state_info=work.obs[state_col_model].astype(str).to_numpy(),
        data_des="gse132188_upto12_official",
    )
    # CoSpar plotting/analysis utilities may expect a low-dim embedding key.
    if ("X_emb" not in ac.obsm) and ("X_pca" in ac.obsm):
        ac.obsm["X_emb"] = np.asarray(ac.obsm["X_pca"][:, :2], dtype=np.float64)

    ac = cs.tmap.infer_Tmap_from_state_info_alone(
        ac,
        initial_time_points=[str(source_time)],
        later_time_point=str(target_time),
        max_iter_N=[1, 2],
        compute_new=True,
        CoSpar_KNN=20,
        smooth_array=[15, 10, 5],
    )

    # Use CoSpar-inferred transition_map directly: source cells -> target cells.
    T = ac.uns["transition_map"]  # sparse matrix, shape=(n_source, n_target)
    src_idx = np.asarray(ac.uns["Tmap_cell_id_t1"], dtype=np.int64)
    tgt_idx = np.asarray(ac.uns["Tmap_cell_id_t2"], dtype=np.int64)
    tgt_labels = ac.obs.iloc[tgt_idx]["state_info"].astype(str).to_numpy()

    P = np.zeros((T.shape[0], len(TARGET_CLASSES)), dtype=np.float64)
    for j, cls in enumerate(TARGET_CLASSES):
        m = tgt_labels == cls
        if np.any(m):
            P[:, j] = np.asarray(T[:, m].sum(axis=1), dtype=np.float64).reshape(-1)
    source_ids = np.array(ac.obs_names[src_idx], dtype=str)
    P = np.clip(P, 1e-9, np.inf)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return {str(cid): P[i] for i, cid in enumerate(source_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_h5", default="/Users/wanghongye/python/scLineagetracer/GSE132188/processed/GSE132188_DeepLineage_all_generated_sequences.h5")
    parser.add_argument("--index_csv", default="/Users/wanghongye/python/scLineagetracer/GSE132188/processed/GSE132188_DeepLineage_all_generated_index.csv")
    parser.add_argument("--h5ad", default="/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/processed_norm_log_hvg1000.h5ad")
    parser.add_argument("--model_dir", default="/Users/wanghongye/python/scLineagetracer/classification/GSE132188/saved_models")
    parser.add_argument("--out_root", default="/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc")
    parser.add_argument("--run_cospar", type=int, default=0, help="1 to run official CoSpar (slower)")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_root, "benchmark_upto_12_5_official")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "cospar_data"))

    te_idx, y_true, p_sc, classes = load_sclineage_upto12_probs(args.seq_h5, args.index_csv, args.model_dir)
    if list(classes) != TARGET_CLASSES:
        raise RuntimeError(f"Unexpected class order in sequence H5: {classes}")

    idx_df = pd.read_csv(args.index_csv)
    te_rows = idx_df.iloc[te_idx]
    source_ids = te_rows["origin"].astype(str).to_numpy()

    adata = ad.read_h5ad(args.h5ad)
    adata.obs["day"] = adata.obs["day"].astype(float)
    raw_state = adata.obs["clusters_fig6_fine_final"].astype(str).to_numpy()
    state_model = np.array([STATE_CANON_MAP.get(x, x) for x in raw_state], dtype=object)
    adata.obs["state_model"] = pd.Categorical(state_model)
    adata.obs_names = pd.Index([str(x) for x in adata.obs_names])

    adata_sub = prepare_subset(
        adata=adata,
        source_ids=source_ids,
        time_col="day",
        state_col_model="state_model",
        source_time=12.5,
        target_time=15.5,
        mid_times=(13.5, 14.5),
        mid_cap_each=3000,
    )

    curves = {"scLineagetracer": eval_multiclass(y_true, p_sc)}
    method_err = []

    try:
        m = run_cellrank_multiclass(adata_sub, "day", "state_model", 12.5, 15.5)
        p = extract_source_matrix_by_ids(m, source_ids)
        curves["CellRank"] = eval_multiclass(y_true, p)
    except Exception as e:
        method_err.append(("CellRank", str(e)))

    try:
        m = run_wot_multiclass(adata_sub, "day", "state_model", 12.5, 15.5)
        p = extract_source_matrix_by_ids(m, source_ids)
        curves["WOT"] = eval_multiclass(y_true, p)
    except Exception as e:
        method_err.append(("WOT", str(e)))

    if int(args.run_cospar) == 1:
        try:
            m = run_cospar_multiclass(adata_sub, "day", "state_model", 12.5, 15.5, os.path.join(out_dir, "cospar_data"))
            p = extract_source_matrix_by_ids(m, source_ids)
            curves["CoSpar"] = eval_multiclass(y_true, p)
        except Exception as e:
            method_err.append(("CoSpar", str(e)))

    rows = []
    per_class_rows = []
    for method, c in curves.items():
        rows.append({
            "Setting": "UpTo_12.5",
            "Method": method,
            "AUC_macro": c["AUC_macro"],
            "Accuracy": c["Accuracy"],
            "LogLoss": c["LogLoss"],
            "N_test": int(len(y_true)),
            "Pipeline": "official",
        })
        for cls, a in zip(TARGET_CLASSES, c["AUC_per_class"]):
            per_class_rows.append({"Method": method, "Class": cls, "AUC": float(a)})

    summary = pd.DataFrame(rows).sort_values("AUC_macro", ascending=False)
    per_class = pd.DataFrame(per_class_rows)
    summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    summary[["Method", "Accuracy"]].to_csv(os.path.join(out_dir, "metrics_accuracy_only.csv"), index=False)
    summary[["Method", "AUC_macro"]].to_csv(os.path.join(out_dir, "metrics_auc_macro_only.csv"), index=False)
    per_class.to_csv(os.path.join(out_dir, "metrics_auc_per_class.csv"), index=False)

    curve_rows = []
    for m, c in curves.items():
        for i in range(len(c["fpr_macro"])):
            curve_rows.append({"Method": m, "point_id": i, "fpr_macro": float(c["fpr_macro"][i]), "tpr_macro": float(c["tpr_macro"][i])})
    pd.DataFrame(curve_rows).to_csv(os.path.join(out_dir, "roc_curve_points_macro.csv"), index=False)

    np.savez(
        os.path.join(out_dir, "benchmark_probs.npz"),
        te_idx=te_idx,
        y_true=y_true,
        p_sclineagetracer=p_sc,
    )

    base = plot_dual(curves, out_dir)

    with open(os.path.join(out_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("Official method run log (GSE132188 UpTo_12.5)\n")
        if method_err:
            f.write("Failed methods:\n")
            for m, e in method_err:
                f.write(f"- {m}: {e}\n")
        else:
            f.write("All methods succeeded.\n")

    print(f"[DONE] Output folder: {out_dir}")
    print(f"[DONE] ROC full: {base}_full.png/.pdf")
    print(f"[DONE] ROC clean: {base}_clean.png/.pdf")
    print(summary.to_string(index=False))
    if method_err:
        print("[WARN] Some methods failed:")
        for m, e in method_err:
            print(f"- {m}: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
