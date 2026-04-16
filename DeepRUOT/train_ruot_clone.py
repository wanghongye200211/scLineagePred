# -*- coding: utf-8 -*-
"""train_ruot_clone.py

DeepRUOT: eval-only (生成 sde_point / sde_weight)，支持减少输入点数避免 OOM。

你只需要改一个地方：
    CONFIG_PATH = "/path/to/your.yaml"

然后直接运行：
    python train_ruot_clone.py

输出（写到训练实验目录 exp_dir）：
    sde_point_{run}.npy        float32  (T, N, D)
    sde_weight_{run}.npy       float32  (T, N, 1)
    sde_t0_row_id_{run}.npy    int64    (N,)   # 本次 eval 实际用到的 t0 行号（对应 input_csv reset_index 后的行号）
    sde_point_meta_{run}.json  记录本次 eval 的关键参数

说明
----
- 本脚本**不训练**，只加载 exp_dir 下已经训练好的权重：
    - model_final
    - score_model_final（如果不存在则尝试 score_model）
- SDE 积分用 Euler–Maruyama，每一步都会 detach，避免 MPS/CUDA 因跨步计算图累积而 OOM。
- 评估参数（如 num_points / batch_size / dt / num_runs / seed）建议写进 YAML 的 `eval:` 段落；
  如果你的 YAML 没有 `eval:`，脚本会用默认值。

建议在 YAML 里加上（可选）：

```yaml
# ... 你原来的配置不动

eval:
  # 为了避免 OOM：只抽 t0 的一部分点
  num_points: 2000        # 不写/写 null 表示用全部 t0

  # 仍然 OOM 就调小 batch_size
  batch_size: 200

  # SDE 步长（与旧 eval.py 默认一致）
  dt: 0.1

  # sigma==0 时默认 1 次；sigma>0 默认 5 次；你也可以显式指定
  num_runs: 5

  # 随机种子（影响抽点和随机噪声）
  seed: 2026

  # 如果你的数据列名不是 samples/x1,x2,...，可改：
  time_col: samples
  x_prefix: x
```

"""

from __future__ import annotations

import os
import sys
import math
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from numpy.lib.format import open_memmap


# ================== 只需要改这里 ==================
CONFIG_PATH = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/config/GSE132188.yaml"
# ==================================================


# ------------------------- path / imports -------------------------
# 允许你从任意工作目录 `python train_ruot_clone.py` 也能 import DeepRUOT
_THIS_FILE = os.path.abspath(__file__)
_DEEPRUOT_DIR = os.path.dirname(_THIS_FILE)               # .../DeepRUOT
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEEPRUOT_DIR, ".."))
for _p in (_PROJECT_ROOT, _DEEPRUOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DeepRUOT.models import FNet, scoreNet2


# ------------------------- helpers -------------------------

def _expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _sorted_feature_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]

    def _key(c: str) -> int:
        tail = c[len(prefix) :]
        return int(tail) if tail.isdigit() else 10**9

    cols.sort(key=_key)
    return cols


def _load_config(config_path: str) -> Dict:
    """优先使用 DeepRUOT.utils.load_and_merge_config（如果可用），失败则 yaml.safe_load。"""
    config_path = _expand(config_path)

    # 1) 尝试带默认值合并（更稳）
    try:
        from DeepRUOT.utils import load_and_merge_config

        cfg = load_and_merge_config(config_path)
        if not isinstance(cfg, dict):
            raise TypeError("load_and_merge_config did not return dict")
        return cfg
    except Exception as e:
        print(f"[warn] load_and_merge_config failed, fallback to yaml.safe_load: {type(e).__name__}: {e}")

    # 2) 直接读 YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a dict")
    return cfg


def _infer_exp_dir(cfg: Dict) -> str:
    """从 YAML 推断训练实验目录 exp_dir（里面应当有 model_final / score_model_final）。"""
    exp = cfg.get("exp", {}) if isinstance(cfg.get("exp", {}), dict) else {}

    # 显式指定最优先
    for k in ("exp_dir", "load_dir", "ckpt_dir"):
        if isinstance(exp.get(k), str) and exp.get(k).strip():
            p = _expand(exp[k])
            if os.path.isdir(p):
                return p

    output_dir = exp.get("output_dir")
    name = exp.get("name")

    if isinstance(output_dir, str) and output_dir.strip() and isinstance(name, str) and name.strip():
        cand = _expand(os.path.join(output_dir, name))
        if os.path.isdir(cand):
            return cand

    # 兜底：扫 output_dir 下面含 model_final 的子目录，取最新
    if isinstance(output_dir, str) and output_dir.strip():
        od = _expand(output_dir)
        if os.path.isdir(od):
            cands = []
            for sub in os.listdir(od):
                p = os.path.join(od, sub)
                if not os.path.isdir(p):
                    continue
                if os.path.isfile(os.path.join(p, "model_final")):
                    cands.append(p)
            if cands:
                # 优先同名
                if isinstance(name, str) and name.strip():
                    for p in cands:
                        if os.path.basename(p) == name:
                            return p
                cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return cands[0]

    raise FileNotFoundError(
        "无法从 YAML 推断 exp_dir。\n"
        "请在 YAML 里写 exp.exp_dir（推荐），或确保 exp.output_dir/exp.name 能定位到训练输出目录。"
    )


def _infer_input_csv(cfg: Dict) -> str:
    data = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}

    # 兼容几种可能的 key
    for k in ("file_path", "input_csv", "csv_path"):
        if isinstance(data.get(k), str) and data.get(k).strip():
            fp = _expand(data[k])
            if os.path.isabs(fp) and os.path.isfile(fp):
                return fp
            # 相对路径：尽量按 DeepRUOT.constants.DATA_DIR 解析
            try:
                from DeepRUOT.constants import DATA_DIR

                cand = _expand(os.path.join(DATA_DIR, data[k]))
                if os.path.isfile(cand):
                    return cand
            except Exception:
                pass
            # 再兜底：按当前工作目录
            cand2 = _expand(data[k])
            if os.path.isfile(cand2):
                return cand2

    raise FileNotFoundError(
        "无法从 YAML 推断输入 CSV。\n"
        "请在 YAML 的 data.file_path（或 data.input_csv）里写你的 ruot_input_*.csv 路径。"
    )


def _load_models(cfg: Dict, exp_dir: str, device: torch.device) -> Tuple[FNet, scoreNet2]:
    mc = cfg["model"]

    f_net = FNet(
        in_out_dim=mc["in_out_dim"],
        hidden_dim=mc["hidden_dim"],
        n_hiddens=mc["n_hiddens"],
        activation=mc["activation"],
    ).to(device)

    sf2m = scoreNet2(
        in_out_dim=mc["in_out_dim"],
        hidden_dim=mc["score_hidden_dim"],
        activation=mc["activation"],
    ).float().to(device)

    model_path = os.path.join(exp_dir, "model_final")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_final not found: {model_path}")
    f_net.load_state_dict(torch.load(model_path, map_location=device))

    score_path = os.path.join(exp_dir, "score_model_final")
    if not os.path.isfile(score_path):
        score_path = os.path.join(exp_dir, "score_model")
    if not os.path.isfile(score_path):
        raise FileNotFoundError(
            f"score_model_final or score_model not found in exp_dir={exp_dir}"
        )
    sf2m.load_state_dict(torch.load(score_path, map_location=device))

    f_net.eval()
    sf2m.eval()

    # eval-only：不更新参数
    for p in f_net.parameters():
        p.requires_grad_(False)
    for p in sf2m.parameters():
        p.requires_grad_(False)

    return f_net, sf2m


def _euler_sdeint_detach(
    sde,
    initial_state: Tuple[torch.Tensor, torch.Tensor],
    dt: float,
    ts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Euler–Maruyama with per-step detach（防 OOM 关键点）"""
    device = initial_state[0].device
    ts_list = [float(x) for x in ts.detach().cpu().tolist()]

    t0 = ts_list[0]
    tf = ts_list[-1]

    z, lnw = initial_state
    z = z.detach()
    lnw = lnw.detach()

    current_time = float(t0)
    next_out = 0

    out_z: List[torch.Tensor] = []
    out_lnw: List[torch.Tensor] = []

    while current_time <= tf + 1e-8:
        if current_time >= ts_list[next_out] - 1e-8:
            out_z.append(z.detach())
            out_lnw.append(lnw.detach())
            next_out += 1
            if next_out >= len(ts_list):
                break

        t_tensor = torch.tensor([current_time], device=device)

        f_z, f_lnw = sde.f(t_tensor, (z, lnw))

        # diagonal diffusion
        noise_z = torch.randn_like(z) * math.sqrt(dt)
        g_z = sde.g(t_tensor, z)

        z = (z + f_z * dt + g_z * noise_z).detach()
        lnw = (lnw + f_lnw * dt).detach()

        current_time += dt

    # pad
    while len(out_z) < len(ts_list):
        out_z.append(z.detach())
        out_lnw.append(lnw.detach())

    return torch.stack(out_z, dim=0), torch.stack(out_lnw, dim=0)


# ------------------------- core -------------------------

def run_eval_only(config_path: str) -> None:
    cfg = _load_config(config_path)

    # ---- eval settings ----
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}

    time_col = str(eval_cfg.get("time_col", "samples"))
    x_prefix = str(eval_cfg.get("x_prefix", "x"))

    dt = float(eval_cfg.get("dt", 0.1))
    batch_size = int(eval_cfg.get("batch_size", 200))
    seed = int(eval_cfg.get("seed", 2026))

    # num_points: None / null 表示用全部 t0
    num_points = eval_cfg.get("num_points", None)
    if num_points in ("None", "null", "NULL", "none", ""):
        num_points = None
    if num_points is not None:
        num_points = int(num_points)
        if num_points <= 0:
            num_points = None

    # 设备
    device_str = str(cfg.get("device", "cpu"))
    device = torch.device(device_str)

    # sigma / use_mass / num_runs
    sigma = float(cfg.get("score_train", {}).get("sigma", 0.0))
    use_mass = bool(cfg.get("use_mass", True))

    default_runs = 1 if sigma == 0 else 5
    num_runs = int(eval_cfg.get("num_runs", default_runs))
    if num_runs <= 0:
        num_runs = default_runs

    # ---- paths ----
    exp_dir = _infer_exp_dir(cfg)
    input_csv = _infer_input_csv(cfg)

    print("\n================ DeepRUOT eval-only (sde_point) ================")
    print(f"config   : {config_path}")
    print(f"exp_dir  : {exp_dir}")
    print(f"input_csv: {input_csv}")
    print(f"device   : {device}")
    print(f"sigma    : {sigma}")
    print(f"use_mass : {use_mass}")
    print(f"dt       : {dt}")
    print(f"batch    : {batch_size}")
    print(f"num_runs : {num_runs}")
    print(f"num_points(t0 subset): {num_points}")
    print("===============================================================\n")

    # ---- load data ----
    df = pd.read_csv(input_csv)
    df = df.reset_index(drop=True)

    if time_col not in df.columns:
        raise KeyError(f"Missing time column '{time_col}' in CSV: {input_csv}")

    x_cols = _sorted_feature_cols(df, x_prefix)
    if not x_cols:
        raise KeyError(
            f"No feature columns found with prefix '{x_prefix}' in CSV. "
            f"(e.g. expected {x_prefix}1,{x_prefix}2,...)"
        )

    times = sorted(df[time_col].unique().tolist())
    t0 = times[0]

    idx0_all = df.index[df[time_col] == t0].to_numpy(dtype=np.int64)
    x0_all = df.loc[idx0_all, x_cols].to_numpy(dtype=np.float32)

    N0 = x0_all.shape[0]
    D = x0_all.shape[1]
    T = len(times)

    # ---- load models ----
    f_net, sf2m = _load_models(cfg, exp_dir, device)

    # ---- define SDE ----
    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g_net, score_model, sigma_: float):
            super().__init__()
            self.drift = ode_drift
            self.g_net = g_net
            self.score = score_model
            self.sigma = float(sigma_)

        def f(self, t, y):
            z, lnw = y
            drift = self.drift(t, z)
            dlnw = self.g_net(t, z)

            # 保留 t 的梯度信息（和旧 eval.py 一致）
            num = z.shape[0]
            t = t.expand(num, 1)

            if self.sigma == 0:
                return drift, dlnw

            # 关键：score 的梯度项（保持 create_graph=True 原逻辑）
            return drift + self.score.compute_gradient(t, z), dlnw

        def g(self, t, z):
            return torch.ones_like(z) * self.sigma

    sde = SDE(f_net.v_net, f_net.g_net, sf2m, sigma)

    ts_points = torch.tensor(times, dtype=torch.float32, device=device)

    # ---- main loop ----
    os.makedirs(exp_dir, exist_ok=True)

    for run_idx in range(num_runs):
        # 每次 run 不同随机种子（噪声不同）
        run_seed = seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        # 抽 t0 子集（每次 run 可复现）
        if num_points is None or num_points >= N0:
            chosen_local = np.arange(N0, dtype=np.int64)
        else:
            rng = np.random.default_rng(run_seed)
            chosen_local = rng.choice(N0, size=num_points, replace=False).astype(np.int64)

        idx0 = idx0_all[chosen_local]                # (N,)
        x0 = x0_all[chosen_local]                    # (N,D)
        N = x0.shape[0]

        # 输出文件
        out_point = os.path.join(exp_dir, f"sde_point_{run_idx}.npy")
        out_weight = os.path.join(exp_dir, f"sde_weight_{run_idx}.npy")
        out_rowid = os.path.join(exp_dir, f"sde_t0_row_id_{run_idx}.npy")
        out_meta = os.path.join(exp_dir, f"sde_point_meta_{run_idx}.json")

        print(f"[run {run_idx}] T={T} N={N} D={D} -> {out_point}")

        # 用 memmap 直接写 .npy，避免内存峰值
        point_mm = open_memmap(out_point, mode="w+", dtype=np.float32, shape=(T, N, D))
        weight_mm = open_memmap(out_weight, mode="w+", dtype=np.float32, shape=(T, N, 1))

        # 分 batch 积分
        for st in range(0, N, batch_size):
            ed = min(st + batch_size, N)
            xb = torch.from_numpy(x0[st:ed]).to(device)

            # 初始 lnw：对当前子集 N 做归一化
            lnw0 = torch.log(torch.ones((ed - st, 1), device=device) / float(N))

            z_traj, lnw_traj = _euler_sdeint_detach(sde, (xb, lnw0), dt=dt, ts=ts_points)

            z_np = z_traj.detach().cpu().numpy().astype(np.float32)          # (T,B,D)
            if use_mass:
                w_np = torch.exp(lnw_traj).detach().cpu().numpy().astype(np.float32)  # (T,B,1)
            else:
                w_np = np.ones((T, ed - st, 1), dtype=np.float32)

            point_mm[:, st:ed, :] = z_np
            weight_mm[:, st:ed, :] = w_np

            point_mm.flush()
            weight_mm.flush()

            # 尽量释放显存/统一内存
            del xb, lnw0, z_traj, lnw_traj
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        # 保存 row_id
        np.save(out_rowid, idx0.astype(np.int64))

        # 保存 meta
        meta = {
            "config_path": _expand(config_path),
            "exp_dir": _expand(exp_dir),
            "input_csv": _expand(input_csv),
            "time_col": time_col,
            "x_prefix": x_prefix,
            "x_cols": x_cols,
            "times": times,
            "sigma": sigma,
            "use_mass": use_mass,
            "dt": dt,
            "batch_size": batch_size,
            "num_points": None if num_points is None else int(num_points),
            "N0_all_t0": int(N0),
            "N_used": int(N),
            "D": int(D),
            "T": int(T),
            "run_idx": int(run_idx),
            "seed": int(run_seed),
            "outputs": {
                "sde_point": os.path.basename(out_point),
                "sde_weight": os.path.basename(out_weight),
                "sde_t0_row_id": os.path.basename(out_rowid),
            },
        }
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[run {run_idx}] saved: sde_point / sde_weight / row_id / meta")

    print("\n[done] eval-only finished.")


# ------------------------- local entry -------------------------

if __name__ == "__main__":
    run_eval_only(CONFIG_PATH)
