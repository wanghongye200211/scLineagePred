# -*- coding: utf-8 -*-
"""
All-timepoint benchmark for remaining 3 datasets.

Datasets:
- GSE114412: UpTo_0..UpTo_4
- GSE175634: Obs_Day1/3/5/7/11
- GSE99915 : Obs_Day9/12/15/21

For each setting, run official-method comparison (no proxy) and output:
- ROC full/clean
- metrics_summary.csv / accuracy / auc CSV
- run_log.txt

Also export merged summary per dataset.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import compare_roc_official_remaining3 as core


def maybe_make_binary_2d(dataset):
    """
    For binary datasets, generate final 2D official-method comparison plots
    from benchmark_probs.npz after all settings finish.
    """
    if dataset not in {"GSE175634", "GSE99915", "GSE140802"}:
        return
    try:
        import plot_binary_official_methods_2d as plot2d
        plot2d.run_dataset(dataset)
        print(f"[DONE] 2D plots generated for {dataset}")
    except Exception as e:
        print(f"[WARN] 2D plotting skipped for {dataset}: {e}")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def get_probs_var(model, loader, device):
    model.eval()
    probs = []
    for x, _, lengths in loader:
        logits = model(x.to(device), lengths.to(device))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def get_prob_pos_fix(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        logits = model(x.to(device))
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs, axis=0)


def gen_meta_114412_all(root_out):
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed/GSE114412_all_generated_sequences.h5"
    idx_csv = "/Users/wanghongye/python/scLineagetracer/GSE114412/processed/GSE114412_all_generated_index.csv"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE114412/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)
        classes = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["classes"][:]]
        time_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["time_labels"][:]]

    seed = 2026
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=seed, stratify=y)
    rel_val = 0.1 / (1.0 - 0.1)
    _tr2, _va = train_test_split(tr_idx, test_size=rel_val, random_state=seed, stratify=y[tr_idx])

    df = pd.read_csv(idx_csv)
    device = pick_device()
    d = X.shape[2]
    c = len(classes)

    metas = []
    for k in range(len(time_labels) - 1):
        setting = f"UpTo_{time_labels[k]}"
        ds = core.SeqDatasetVarLen(X, y, te_idx.astype(np.int64), keep_len=k + 1)
        dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=core.collate_pad)

        m1 = core.BiLSTMVar(d, 256, 2, 0.10, c).to(device)
        m2 = core.RNNVar(d, 256, 2, 0.10, c).to(device)
        m3 = core.TransformerVar(d, 256, 2, 0.10, 4, c).to(device)

        m1.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_BiLSTM_s2026.pth"), map_location=device))
        m2.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_RNN_s2026.pth"), map_location=device))
        m3.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_Trans_s2026.pth"), map_location=device))

        p1 = get_probs_var(m1, dl, device)
        p2 = get_probs_var(m2, dl, device)
        p3 = get_probs_var(m3, dl, device)
        with open(os.path.join(model_dir, f"{setting}_Stacking_s2026.pkl"), "rb") as f:
            stk = pickle.load(f)
        p_stack = stk.predict_proba(np.concatenate([p1, p2, p3], axis=1))

        source_ids = df.iloc[te_idx][f"id_t{k}"].astype(str).to_numpy()
        mid_times = [str(x) for x in time_labels[k + 1 : -1]]

        metas.append(
            {
                "dataset": "GSE114412",
                "task": "multiclass",
                "setting": setting,
                "target_classes": classes,
                "y_true": y[te_idx],
                "p_sc": p_stack,
                "source_ids": source_ids,
                "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE114412/preprocess_final/processed_norm_log_hvg1000.h5ad",
                "out_dir": os.path.join(root_out, setting),
                "state_col_raw": "Assigned_cluster",
                "time_col_raw": "_week",
                "source_time_raw": str(time_labels[k]),
                "target_time_raw": str(time_labels[-1]),
                "mid_times_raw": mid_times,
                "target_cap_each": 1200,
                "source_cap": 1200,
                "mid_cap_each": 1200,
                "time_map": {str(t): float(t) for t in time_labels},
            }
        )

    return metas


def gen_meta_175634_all(root_out):
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_sequences.h5"
    idx_csv = "/Users/wanghongye/python/scLineagetracer/GSE175634/processed/GSE175634_CMvsCF_all_generated_index.csv"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)

    df = pd.read_csv(idx_csv)
    device = pick_device()
    d = X.shape[2]
    T = X.shape[1]

    conf = [
        ("Obs_Day1", 7, list(range(2, T)), 1, "day1", ["day3", "day5", "day7", "day11"]),
        ("Obs_Day3", 999, [T - 4, T - 3, T - 2, T - 1], 2, "day3", ["day5", "day7", "day11"]),
        ("Obs_Day5", 123, [T - 3, T - 2, T - 1], 3, "day5", ["day7", "day11"]),
        ("Obs_Day7", 42, [T - 2, T - 1], 4, "day7", ["day11"]),
        ("Obs_Day11", 2024, [T - 1], 5, "day11", []),
    ]

    metas = []
    for setting, seed, mask, src_idx, src_time, mids in conf:
        idx = np.arange(len(y))
        tr_idx, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
        _va, te_idx = train_test_split(tmp, test_size=0.5, random_state=seed)
        _ = tr_idx

        ds = core.SeqDatasetMask(X, y, te_idx.astype(np.int64), mask_t=mask)
        dl = DataLoader(ds, batch_size=512, shuffle=False)

        m1 = core.LSTMFix(d, 256, 2, 0.3).to(device)
        m2 = core.RNNFix(d, 256, 2, 0.3).to(device)
        m3 = core.TransformerFix(d, 256, 2, 0.3, 4).to(device)

        m1.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_BiLSTM_s{seed}.pth"), map_location=device))
        m2.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_RNN_s{seed}.pth"), map_location=device))
        m3.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_Trans_s{seed}.pth"), map_location=device))

        p1 = get_prob_pos_fix(m1, dl, device)
        p2 = get_prob_pos_fix(m2, dl, device)
        p3 = get_prob_pos_fix(m3, dl, device)
        with open(os.path.join(model_dir, f"{setting}_Stacking_s{seed}.pkl"), "rb") as f:
            stk = pickle.load(f)
        p_pos = stk.predict_proba(np.c_[p1, p2, p3])[:, 1]

        source_ids = df.iloc[te_idx][f"id_t{src_idx}"].astype(str).to_numpy()

        metas.append(
            {
                "dataset": "GSE175634",
                "task": "binary",
                "setting": setting,
                "target_classes": ["CM", "CF"],
                "positive_class": "CF",
                "y_true": y[te_idx],
                "p_sc": p_pos,
                "source_ids": source_ids,
                "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE175634/preprocess_final/processed_norm_log_hvg1000.h5ad",
                "out_dir": os.path.join(root_out, setting),
                "state_col_raw": "type",
                "time_col_raw": "diffday",
                "source_time_raw": src_time,
                "target_time_raw": "day15",
                "mid_times_raw": mids,
                "target_cap_each": 1200,
                "source_cap": 0,
                "mid_cap_each": 1200,
                "time_map": {
                    "day0": 0.0,
                    "day1": 1.0,
                    "day3": 3.0,
                    "day5": 5.0,
                    "day7": 7.0,
                    "day11": 11.0,
                    "day15": 15.0,
                },
            }
        )

    return metas


def gen_meta_99915_all(root_out):
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE99915/processed/GSE99915_DeepLineage_Masked_sequences.h5"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        y = np.asarray(f["y"], dtype=np.int64)
        indices = np.asarray(f["indices"], dtype=np.int64)

    device = pick_device()
    d = X.shape[2]

    conf = [
        ("Obs_Day9", 999, [2, 3, 4, 5], 1, "Day9", ["Day12", "Day15", "Day21"]),
        ("Obs_Day12", 123, [3, 4, 5], 2, "Day12", ["Day15", "Day21"]),
        ("Obs_Day15", 42, [4, 5], 3, "Day15", ["Day21"]),
        ("Obs_Day21", 2024, [5], 4, "Day21", []),
    ]
    idx_to_time = {
        0: "Day6",
        1: "Day9",
        2: "Day12",
        3: "Day15",
        4: "Day21",
        5: "Day28",
    }

    metas = []
    for setting, seed, mask, src_idx, src_time, mids in conf:
        idx = np.arange(len(y))
        tr_idx, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
        _va, te_idx = train_test_split(tmp, test_size=0.5, random_state=seed)
        _ = tr_idx

        ds = core.SeqDatasetMask(X, y, te_idx.astype(np.int64), mask_t=mask)
        dl = DataLoader(ds, batch_size=512, shuffle=False)

        m1 = core.LSTMFix(d, 256, 2, 0.3).to(device)
        m2 = core.RNNFix(d, 256, 2, 0.3).to(device)
        m3 = core.TransformerFix(d, 256, 2, 0.3, 4).to(device)

        m1.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_BiLSTM_s{seed}.pth"), map_location=device))
        m2.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_RNN_s{seed}.pth"), map_location=device))
        m3.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_Trans_s{seed}.pth"), map_location=device))

        p1 = get_prob_pos_fix(m1, dl, device)
        p2 = get_prob_pos_fix(m2, dl, device)
        p3 = get_prob_pos_fix(m3, dl, device)
        with open(os.path.join(model_dir, f"{setting}_Stacking_s{seed}.pkl"), "rb") as f:
            stk = pickle.load(f)
        p_pos = stk.predict_proba(np.c_[p1, p2, p3])[:, 1]

        src_ids = []
        src_times = []
        for i in te_idx:
            col = src_idx
            sid = None
            st = None
            while col >= 0:
                v = int(indices[i, col])
                if v >= 0:
                    sid = str(v)
                    st = idx_to_time.get(int(col), None)
                    break
                col -= 1
            src_ids.append(sid)
            src_times.append(st)

        metas.append(
            {
                "dataset": "GSE99915",
                "task": "binary",
                "setting": setting,
                "target_classes": ["Failed", "Reprogrammed"],
                "positive_class": "Reprogrammed",
                "y_true": y[te_idx],
                "p_sc": p_pos,
                "source_ids": np.array(src_ids, dtype=object),
                "source_time_per_sample": np.array(src_times, dtype=object),
                "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE99915/preprocess_final/processed_norm_log_hvg1000.h5ad",
                "out_dir": os.path.join(root_out, setting),
                "state_col_raw": "state_info",
                "time_col_raw": "time_info",
                "source_time_raw": src_time,
                "target_time_raw": "Day28",
                "mid_times_raw": mids,
                "target_cap_each": 1200,
                "source_cap": 0,
                "mid_cap_each": 1200,
                "time_map": {
                    "Day6": 6.0,
                    "Day9": 9.0,
                    "Day12": 12.0,
                    "Day15": 15.0,
                    "Day21": 21.0,
                    "Day28": 28.0,
                },
            }
        )

    return metas


def _infer_reverse_from_index_csv(index_csv):
    if not os.path.isfile(index_csv):
        return False
    try:
        df = pd.read_csv(index_csv, usecols=["samples_order"])
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


def _safe_time_to_float(v, default_rank):
    try:
        return float(str(v))
    except Exception:
        return float(default_rank)


def gen_meta_140802_all(root_out):
    idx_csv = "/Users/wanghongye/python/scLineagetracer/GSE140802/processed/GSE140802_DeepLineage_index.csv"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/saved_models"
    idx_df = pd.read_csv(idx_csv)
    idx_bin = idx_df[idx_df["label_str"].isin(["Monocyte", "Neutrophil"])].reset_index(drop=True)

    conf = [
        ("Day2_Only", 42, "idx_t0", "2", ["4"]),
        ("Day2_Day4", 2024, "idx_t1", "4", []),
        ("All_Days", 2024, "idx_t2", "6", []),
    ]

    metas = []
    for setting, seed, src_col, src_time, mids in conf:
        cache_path = os.path.join(model_dir, f"{setting}_TestCache_s{seed}.npz")
        cache = np.load(cache_path, allow_pickle=True)
        te_idx = cache["te_idx"].astype(np.int64)
        y_true = cache["y_true"].astype(np.int64)
        p_sc = np.asarray(cache["p_stack"], dtype=np.float64)
        source_ids = idx_bin.iloc[te_idx][src_col].astype(str).to_numpy()

        metas.append(
            {
                "dataset": "GSE140802",
                "task": "binary",
                "setting": setting,
                "target_classes": ["Neutrophil", "Monocyte"],
                "positive_class": "Monocyte",
                "y_true": y_true,
                "p_sc": p_sc,
                "source_ids": source_ids,
                "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE140802/preprocess_final/processed_norm_log_hvg1000.h5ad",
                "out_dir": os.path.join(root_out, setting),
                "state_col_raw": "state_info",
                "time_col_raw": "time_info",
                "source_time_raw": src_time,
                "target_time_raw": "6",
                "mid_times_raw": mids,
                "target_cap_each": 1500,
                "source_cap": 0,
                "mid_cap_each": 1500,
                "time_map": {"2": 2.0, "4": 4.0, "6": 6.0},
            }
        )
    return metas


def gen_meta_132188_all(root_out):
    seq_h5 = "/Users/wanghongye/python/scLineagetracer/GSE132188/processed/GSE132188_DeepLineage_all_generated_sequences.h5"
    idx_csv = "/Users/wanghongye/python/scLineagetracer/GSE132188/processed/GSE132188_DeepLineage_all_generated_index.csv"
    model_dir = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/saved_models"

    with h5py.File(seq_h5, "r") as f:
        X = np.asarray(f["X"], dtype=np.float32)
        time_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["time_labels"][:]]

    idx_df = pd.read_csv(idx_csv)
    reverse_order = _infer_reverse_from_index_csv(idx_csv)
    if reverse_order:
        X = X[:, ::-1, :].copy()

    # Keep class encoding aligned with class_132188.py:
    # use label_str -> fixed class order mapping.
    target_classes = ["Alpha", "Beta", "Delta", "Epsilon"]
    if "label_str" not in idx_df.columns:
        raise KeyError("Missing required column 'label_str' in GSE132188 index CSV.")

    labels_all = idx_df["label_str"].astype(str).to_numpy()
    keep_mask = np.isin(labels_all, np.array(target_classes, dtype=object))
    if int(keep_mask.sum()) <= 0:
        raise RuntimeError("No rows remain after filtering to target classes for GSE132188.")

    X = X[keep_mask]
    idx_df = idx_df.loc[keep_mask].reset_index(drop=True)
    labels = labels_all[keep_mask]
    label_to_y = {c: i for i, c in enumerate(target_classes)}
    y = np.array([label_to_y[s] for s in labels], dtype=np.int64)

    seed = 2026
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=seed, stratify=y)
    rel_val = 0.1 / (1.0 - 0.1)
    _tr2, _va = train_test_split(tr_idx, test_size=rel_val, random_state=seed, stratify=y[tr_idx])

    device = pick_device()
    d = X.shape[2]
    c = len(target_classes)
    T = len(time_labels)
    if T <= 1:
        raise RuntimeError(f"Unexpected number of time points in GSE132188: T={T}")

    settings = []
    for k in range(T - 1):
        settings.append(
            {
                "name": f"UpTo_{time_labels[k]}",
                "keep_len": int(k + 1),
                "src_time": str(time_labels[k]),
                "mid_times": [str(x) for x in time_labels[k + 1 : -1]],
            }
        )
    settings.append(
        {
            "name": f"All_{time_labels[-1]}",
            "keep_len": int(T),
            "src_time": str(time_labels[-1]),
            "mid_times": [],
        }
    )

    metas = []
    for item in settings:
        setting = item["name"]
        keep_len = int(item["keep_len"])
        src_time = str(item["src_time"])
        mids = list(item["mid_times"])

        k = keep_len - 1
        src_idx_in_index = (T - 1 - k) if reverse_order else k
        src_col = f"id_t{int(src_idx_in_index)}"
        if src_col not in idx_df.columns:
            raise KeyError(f"Missing required source id column for {setting}: {src_col}")

        ds = core.SeqDatasetVarLen(X, y, te_idx.astype(np.int64), keep_len=keep_len)
        dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=core.collate_pad)

        m1 = core.BiLSTMVar(d, 256, 2, 0.10, c).to(device)
        m2 = core.RNNVar(d, 256, 2, 0.10, c).to(device)
        m3 = core.TransformerVar(d, 256, 2, 0.10, 4, c).to(device)
        m1.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_BiLSTM_s2026.pth"), map_location=device))
        m2.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_RNN_s2026.pth"), map_location=device))
        m3.load_state_dict(torch.load(os.path.join(model_dir, f"{setting}_Trans_s2026.pth"), map_location=device))

        p1 = get_probs_var(m1, dl, device)
        p2 = get_probs_var(m2, dl, device)
        p3 = get_probs_var(m3, dl, device)
        with open(os.path.join(model_dir, f"{setting}_Stacking_s2026.pkl"), "rb") as f:
            stk = pickle.load(f)
        p_stack = stk.predict_proba(np.concatenate([p1, p2, p3], axis=1))

        source_ids = idx_df.iloc[te_idx][src_col].astype(str).to_numpy()

        tmap = {}
        for i, tl in enumerate(time_labels):
            tmap[str(tl)] = _safe_time_to_float(tl, default_rank=i)

        metas.append(
            {
                "dataset": "GSE132188",
                "task": "multiclass",
                "setting": setting,
                "target_classes": target_classes,
                "y_true": y[te_idx],
                "p_sc": p_stack,
                "source_ids": source_ids,
                "h5ad": "/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/processed_norm_log_hvg1000.h5ad",
                "out_dir": os.path.join(root_out, setting),
                "state_col_raw": "clusters_fig6_fine_final",
                "state_map": {
                    "Alpha": "Alpha",
                    "Fev+ Alpha": "Alpha",
                    "Beta": "Beta",
                    "Fev+ Beta": "Beta",
                    "Delta": "Delta",
                    "Fev+ Delta": "Delta",
                    "Epsilon": "Epsilon",
                    "Fev+ Epsilon": "Epsilon",
                },
                "time_col_raw": "day",
                "source_time_raw": src_time,
                "target_time_raw": str(time_labels[-1]),
                "mid_times_raw": mids,
                "target_cap_each": 1200,
                "source_cap": 0,
                "mid_cap_each": 1200,
                "time_map": tmap,
            }
        )
    return metas


def run_dataset(dataset, run_cospar):
    if dataset == "GSE114412":
        root_out = "/Users/wanghongye/python/scLineagetracer/classification/GSE114412/roc/benchmark_all_timepoints_official_plus"
        metas = gen_meta_114412_all(root_out)
    elif dataset == "GSE175634":
        root_out = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_all_timepoints_official"
        metas = gen_meta_175634_all(root_out)
    elif dataset == "GSE140802":
        root_out = "/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus"
        metas = gen_meta_140802_all(root_out)
    elif dataset == "GSE132188":
        root_out = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/benchmark_all_timepoints_official_plus"
        metas = gen_meta_132188_all(root_out)
    else:
        root_out = "/Users/wanghongye/python/scLineagetracer/classification/GSE99915/roc_click/benchmark_all_timepoints_official_plus"
        metas = gen_meta_99915_all(root_out)

    core.ensure_dir(root_out)

    merged = []
    for m in metas:
        print(f"[RUN] {dataset} {m['setting']}")
        core.run_one(m, run_cospar=run_cospar)
        s = pd.read_csv(os.path.join(m["out_dir"], "metrics_summary.csv"))
        s.insert(0, "Dataset", dataset)
        merged.append(s)

    all_df = pd.concat(merged, ignore_index=True)
    all_df.to_csv(os.path.join(root_out, "metrics_summary_all_settings.csv"), index=False)

    acc_cols = [c for c in ["Dataset", "Setting", "Method", "Accuracy", "MethodNote", "MissingSourceFallback"] if c in all_df.columns]
    auc_cols = [c for c in ["Dataset", "Setting", "Method", "AUC", "AUC_macro", "MethodNote", "MissingSourceFallback"] if c in all_df.columns]

    all_df[acc_cols].to_csv(os.path.join(root_out, "metrics_accuracy_all_settings.csv"), index=False)
    all_df[auc_cols].to_csv(os.path.join(root_out, "metrics_auc_all_settings.csv"), index=False)

    print(f"[DONE] merged summary: {os.path.join(root_out, 'metrics_summary_all_settings.csv')}")
    maybe_make_binary_2d(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GSE114412", "GSE175634", "GSE99915", "GSE140802", "GSE132188"])
    parser.add_argument("--run_cospar", type=int, default=1, help="kept for compatibility; CoSpar is always forced official.")
    args = parser.parse_args()

    run_dataset(args.dataset, run_cospar=int(args.run_cospar))


if __name__ == "__main__":
    main()
