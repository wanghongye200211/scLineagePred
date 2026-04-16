# -*- coding: utf-8 -*-
"""
decoder_template_multiclass.py

功能（Functions）:
1) 多类别散点图 scatter plot（散点图）:
   - 自动按细胞类型 cell type（类别）拆开
   - 每类单独计算 Pearson r（皮尔逊相关）和 R²（决定系数）
   - 输出 scatter_<class>.png
   - 可选突出点 highlight（突出）+ 可选标注 gene names（基因名）

2) 多类别小提琴图 violin plot（小提琴图）:
   - 每类输出 violin_<class>.png
   - 基因可手动指定（manual）或自动按每类 Top-R² 选择（auto）
   - 处理零膨胀（zero inflation/大量0）可开关：只画 true>0 的细胞

输入（NPZ keys）:
- pred_log: shape (N, G)
- true_log: shape (N, G)
- label (or cell_type): shape (N,)
- gene_names: shape (G,)
- optional clone_id: shape (N,)  (如果想按 clone 求均值)

输出（Outputs）:
- <task>/<OUT_SUBDIR>/scatter_<class>.png
- <task>/<OUT_SUBDIR>/violin_<class>.png
- <task>/<OUT_SUBDIR>/scatter_r2_summary.csv
- <task>/<OUT_SUBDIR>/violin_gene_table_<class>.csv (每类基因表，方便写论文)
"""

# =========================================================
# ===============  EDIT HERE（你只改这里）  =================
# =========================================================
REG_OUT_DIR = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE175634"
TASKS = None                      # None -> 自动扫描包含 test_outputs.npz 的任务目录
NPZ_NAME = "test_outputs.npz"
OUT_SUBDIR = "plots_template_v1"

# ---------- 类别颜色 class colors（颜色放最前，方便调） ----------
# 你可以只写其中几类，其它类会用 tab10 自动补足
CLASS_COLORS = {
    # "Neutrophil": "#4C78A8",
    # "Monocyte":   "#F58518",
    # "Class3":     "#54A24B",
    # "Class4":     "#B279A2",
}
FALLBACK_PALETTE = "tab10"
FALLBACK_COLOR = "#9ecae1"
DIAG_LINE_COLOR = "#6E6E6E"

# ---------- 全局开关 ----------
PLOT_SCATTER = True
PLOT_VIOLIN  = True

# ---------- 聚合模式 aggregation（聚合） ----------
# scatter/violin 都可以选择按 clone 求均值或按 cell
# 若 NPZ 没有 clone_id，会自动退化到 cell
AGG_MODE_SCATTER = "clone"        # "clone" or "cell"
AGG_MODE_VIOLIN  = "cell"         # 通常 violin 看细胞分布，默认 cell 更合理


MIN_SAMPLES_PER_CLASS = 10

# ---------- scatter 筛选（filter/mask） ----------
SCATTER_POSITIVE_ONLY = True      # log1p 一般 True；如果你数据有负数（z-score）改 False
SCATTER_TICK_STEP = 1.5
SCATTER_POINT_SIZE = 60
SCATTER_POINT_ALPHA = 0.72
AXIS_MODE = "auto"                # "auto" or "fixed"
AXIS_FIXED_LIM = (0.0, 1.0)       # AXIS_MODE="fixed" 才生效
AXIS_MARGIN_RATIO = -0.05

# ---------- scatter 突出点（highlight）开关 ----------
HIGHLIGHT_POINTS = False          # 突出点（highlight points）
HIGHLIGHT_LABELS = False             # 突出点 + 标注基因名（gene labels）
# 只要突出点不要名字：HIGHLIGHT_POINTS=True, HIGHLIGHT_LABELS=False

HIGHLIGHT_SELECTION = "manual"    # "manual" | "targets" | "top_r2"
HIGHLIGHT_TOPK = 6                # when selection="top_r2"

# 手动突出基因（global）: 对所有类别都高亮
HIGHLIGHT_GENES_GLOBAL = [
    # "Axl", "Bcl6"
]
# 手动突出基因（按类别分别指定）: 优先级高于 global
#HIGHLIGHT_GENES_BY_CLASS = {
     #"Neutrophil": ["Ampd3","Lgals3","Nfkbia","S100a6"],
     #"Monocyte": [ "Tmem140","Plxna1","Cd300c2" ,"Ccl6","cybb"],
#}
HIGHLIGHT_GENES_BY_CLASS = {
     #"Neutrophil": ["Ampd3","Lgals3","Nfkbia","S100a6"],
     #"Monocyte": [ "Tmem140","Plxna1","Cd300c2" ,"Ccl6","cybb"],
}
#HIGHLIGHT_GENES_BY_CLASS = {
     #"Neutrophil": ["Ampd3","Lgals3","Nfkbia","S100a6"],
     #"Monocyte": [ "Tmem140","Plxna1","Cd300c2" ,"Ccl6","cybb"],
#}
#HIGHLIGHT_GENES_BY_CLASS = {
     #"Neutrophil": ["Ampd3","Lgals3","Nfkbia","S100a6"],
     #"Monocyte": [ "Tmem140","Plxna1","Cd300c2" ,"Ccl6","cybb"],
#}
# 自动 targets 模式：挑选每类真实均值（real mean）最接近这些 target 的基因
HIGHLIGHT_TARGETS = [0.1, 1.0, 1.67, 2.23, 2.5]
HIGHLIGHT_OUTLIER_PERCENTILE = 99.5  # 避免标注极端点

HIGHLIGHT_POINT_SIZE = 220
HIGHLIGHT_EDGE_COLOR = "black"
HIGHLIGHT_EDGE_WIDTH = 0.9

LABEL_FONT_SIZE = 11
LABEL_DX_RATIO = 0.025
LABEL_DY_RATIO = 0.090

# ---------- violin 参数 ----------
import seaborn as sns
VIOLIN_PALETTE = {"Real": "#A6CEE3", "Pred": "#FB9A99"}  # 可调
VIOLIN_INNER = "box"
VIOLIN_LINEWIDTH = 1.0
VIOLIN_CUT = 3              # cut（切割）: 0 会让 KDE 在 min/max 处“平底”
VIOLIN_SCALE = "width"
VIOLIN_BW_ADJUST = 1.0
VIOLIN_GRIDSIZE = 200

# y 轴范围（ylim/纵轴范围）: 用分位数控制显示，避免极端值撑爆
VIOLIN_YLIM_PERCENTILES = (0.0, 100.0)
VIOLIN_YLIM_MARGIN = 0.06
VIOLIN_BOTTOM_EXTRA_RATIO = 0.08
VIOLIN_TOP_HEADROOM_RATIO = 0.40
VIOLIN_BOTTOM_FLOOR = None     # None 不强制；若你想保证含 0 可设 0.0

# 只画真实值 true>0 的细胞（推荐用于零膨胀）
VIOLIN_ONLY_EXPRESSED_TRUE = False
VIOLIN_TOP_K = 15
VIOLIN_GENE_SELECT = "manual"   # "manual" | "top_r2"
#VIOLIN_MANUAL_GENES_BY_CLASS = {
    # "Neutrophil": ["Cybb","Ltf","Mpo","S100a6","Lyz2"],
    # "Monocyte": ["Ly6a" ,"Cadm1"   ,"Lyz2","Ccl6","Pdpn"],
#}
#VIOLIN_MANUAL_GENES_BY_CLASS = {
    # "Reprogrammed": ["Sord","C4bp","Nuak2","Cybrd1","Fmo1","Plcb1"],
    # "Failed": ["Igf2" ,"Ube2c","CellTag.UTR","GFP.CDS","Spp1","Gem"],
#}
#VIOLIN_MANUAL_GENES_BY_CLASS = {
   #  "sc_beta": ["CLPS","NPTX2","FN1","IGFBP5","PDE3A","SULT4A1"],
   # "sc_alpha": ["EGR1" ,"CACNB3","HPGD","CD36","CXCL1","IGFBP7"],
#"sc_ec": ["RET" ,"FEV","GAP43","ADARB2","MAOB","ITM2A"],
#}
VIOLIN_MANUAL_GENES_BY_CLASS = {
     "CM": ["MYH6","MYL4","TTN","SMPX","ACTC1","ACTN2"],
     "CF": ["TECRL" ,"EZR","S100A10","GPC3","CALB2","KRT8"],
}
#VIOLIN_MANUAL_GENES_BY_CLASS = {
#     "Beta": ["Pcsk2","Iapp","Tac1","Fos","Vim","Tmsb15l"],
#   "Alpha": ["Ins2" ,"Nnat","Npy","Col6a3","Iapp","Arhgap36"],
#"Delta": ["Iapp" ,"Mid1ip1","Bambi","Ocrl","Map1b","Mapre3"],
#"Epsilon": ["Neurog3" ,"Tox3","Hist1h2bc","Kcnma1","Fxyd3","Mdk"],
#}
# 表达过滤：真实值>0 的细胞数/比例过低的基因不参与 top_r2 排序
VIOLIN_MIN_POS_CELLS = 10
VIOLIN_MIN_POS_FRAC  = 0.02

# 永久排除基因（排除列表）
EXCLUDE_GENES = {"penk"}        # 全部小写


# =========================================================
# ======================  IMPORTS  =========================
# =========================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


# =========================================================
# ======================  UTILS  ===========================
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-_\.]+", "_", s)
    return s.strip("_")

def decode_arr(a):
    if a is None:
        return None
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        out = []
        for x in a:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8").strip())
            else:
                out.append(str(x).strip())
        return np.array(out, dtype=object)
    return a

def pick_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None

def pearson_r_and_r2_vec(X, Y, eps=1e-12):
    """
    向量化计算每个基因的 Pearson r / R²
    X,Y: (n, G)
    returns: r (G,), r2 (G,)
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my

    vx = np.mean(Xc * Xc, axis=0)
    vy = np.mean(Yc * Yc, axis=0)
    cov = np.mean(Xc * Yc, axis=0)
    denom = np.sqrt(vx * vy)

    r = np.full_like(denom, np.nan, dtype=np.float64)
    ok = denom > eps
    r[ok] = cov[ok] / denom[ok]
    r2 = r * r
    return r.astype(np.float32), r2.astype(np.float32)

def build_color_map(classes, user_map, palette_name="tab10"):
    user_map = dict(user_map) if user_map else {}
    cmap = {}
    try:
        base = plt.get_cmap(palette_name)
        base_cols = [matplotlib.colors.to_hex(base(i)) for i in range(max(len(classes), 10))]
    except Exception:
        base_cols = [FALLBACK_COLOR] * max(len(classes), 10)

    used = set([c.lower() for c in user_map.values()])
    j = 0
    for cls in classes:
        if cls in user_map:
            cmap[cls] = user_map[cls]
        else:
            while j < len(base_cols) and base_cols[j].lower() in used:
                j += 1
            cmap[cls] = base_cols[j] if j < len(base_cols) else FALLBACK_COLOR
            used.add(cmap[cls].lower())
            j += 1
    return cmap

from matplotlib.ticker import FixedLocator, FixedFormatter

def set_violin_right_ticks_4(ax, y_min=0.0, n_ticks=4):
    """
    Right-side y-axis: only show n_ticks ticks, starting from y_min, evenly spaced.
    """
    # Use current ylim
    lo, hi = ax.get_ylim()
    lo = float(lo)
    hi = float(hi)

    start = float(y_min)
    end = hi

    # If start is above end, fallback to current lo
    if start >= end:
        start = lo

    ticks = np.linspace(start, end, n_ticks)

    # Format: keep 2 decimals if needed, else 1
    labels = []
    for t in ticks:
        if abs(t - round(t)) < 1e-6:
            labels.append(str(int(round(t))))
        else:
            labels.append(f"{t:.2f}".rstrip("0").rstrip("."))

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))

    # Show ticks on right, hide left (as per your request "右侧只标记")
    ax.tick_params(axis="y", which="major",
                   left=False, right=True,
                   labelleft=False, labelright=True,
                   direction="out",
                   length=6, width=1.2, colors="black", pad=6)


def compute_clone_means(X, clone_id):
    clone_id = np.asarray(clone_id)
    uniq, inv = np.unique(clone_id, return_inverse=True)
    C = len(uniq)
    G = X.shape[1]
    sums = np.zeros((C, G), dtype=np.float64)
    np.add.at(sums, inv, X.astype(np.float64))
    cnt = np.bincount(inv).astype(np.float64)
    return (sums / np.maximum(cnt, 1.0)[:, None]).astype(np.float32), uniq


def clean_variant_path(out_png: str) -> str:
    root, ext = os.path.splitext(out_png)
    if not ext:
        ext = ".png"
    return f"{root}_clean{ext}"


def strip_plot_text_for_clean(ax, hide_tick_labels: bool = True):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if hide_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

def style_ax(ax, set_y_locator=True):
    ax.grid(False)
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.6)

    # ✅ x-axis: show bottom tick marks + labels
    ax.tick_params(
        axis="x", which="major",
        bottom=True, top=False,
        direction="out",
        length=6, width=1.2, colors="black",
        pad=6
    )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    # ✅ y-axis: left only, tick marks visible
    if set_y_locator:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(
        axis="y", which="major",
        left=True, right=False,
        labelleft=True, labelright=False,
        direction="out",
        length=6, width=1.2, colors="black",
        pad=6
    )
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    ax.minorticks_off()


def list_tasks(root):
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if os.path.exists(os.path.join(root, name, NPZ_NAME)):
            out.append(name)
    return out

def annotate_labels(ax, x, y, names):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return

    xr = max(x.max() - x.min(), 1e-6)
    yr = max(y.max() - y.min(), 1e-6)
    dx = LABEL_DX_RATIO * xr
    dy = LABEL_DY_RATIO * yr

    order = np.argsort(y)
    x = x[order]; y = y[order]
    names = [names[i] for i in order]

    placed_y = []
    for i in range(x.size):
        ha = "left" if (i % 2 == 0) else "right"
        x_text = x[i] + (dx if ha == "left" else -dx)

        k = (i // 2)
        sgn = 1 if (i % 2 == 0) else -1
        y_text = y[i] + (sgn * (k * 0.55) * dy)

        for _ in range(10):
            if all(abs(y_text - yy) >= 0.55 * dy for yy in placed_y):
                break
            y_text += 0.65 * dy
        placed_y.append(y_text)

        ax.text(
            x_text, y_text, str(names[i]),
            fontsize=LABEL_FONT_SIZE, fontweight="bold",
            color="black", ha=ha, va="center",
            zorder=50, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88)
        )

def select_highlight_indices(cls, genes, x, y, r2_gene=None):
    """
    返回要高亮的 gene indices.
    genes: (G,) names
    x,y: gene means arrays (G,)
    r2_gene: (G,) optional, for top_r2 selection
    """
    g2i = {str(g): i for i, g in enumerate(genes)}
    # 1) manual by class
    if HIGHLIGHT_SELECTION == "manual":
        lst = HIGHLIGHT_GENES_BY_CLASS.get(cls, None)
        if lst is None or len(lst) == 0:
            lst = HIGHLIGHT_GENES_GLOBAL
        idx = []
        for g in lst:
            if g in g2i:
                idx.append(g2i[g])
        return np.array(idx, dtype=int)

    # common outlier thresholds
    x_plot = x[np.isfinite(x)]
    y_plot = y[np.isfinite(y)]
    if x_plot.size == 0 or y_plot.size == 0:
        return np.array([], dtype=int)
    x_thr = float(np.percentile(x_plot, HIGHLIGHT_OUTLIER_PERCENTILE))
    y_thr = float(np.percentile(y_plot, HIGHLIGHT_OUTLIER_PERCENTILE))

    valid = np.isfinite(x) & np.isfinite(y)
    if SCATTER_POSITIVE_ONLY:
        valid = valid & (x > 0) & (y > 0)

    # 2) targets
    if HIGHLIGHT_SELECTION == "targets":
        cand = np.where(valid & (x <= x_thr) & (y <= y_thr))[0]
        if cand.size == 0:
            return np.array([], dtype=int)

        selected = []
        used = set()
        for t in HIGHLIGHT_TARGETS:
            dist = np.abs(y[cand] - float(t))
            order = np.argsort(dist)
            pick = None
            for j in order:
                gi = int(cand[j])
                if gi not in used:
                    pick = gi
                    break
            if pick is not None:
                selected.append(pick)
                used.add(pick)
        return np.array(selected, dtype=int)

    # 3) top_r2
    if HIGHLIGHT_SELECTION == "top_r2":
        if r2_gene is None:
            return np.array([], dtype=int)
        cand = np.where(valid)[0]
        if cand.size == 0:
            return np.array([], dtype=int)
        score = np.nan_to_num(r2_gene[cand], nan=-1e9)
        order = np.argsort(-score)
        top = cand[order[:HIGHLIGHT_TOPK]]
        return np.array(top, dtype=int)

    return np.array([], dtype=int)


# =========================================================
# ======================  PLOTS  ===========================
# =========================================================
def plot_scatter_per_class(pred_X, true_X, genes, out_png, title, point_color, cls_name):
    # gene means
    x = pred_X.mean(axis=0)
    y = true_X.mean(axis=0)

    # mask/filter（筛选）
    m = np.isfinite(x) & np.isfinite(y)
    if SCATTER_POSITIVE_ONLY:
        m = m & (x > 0) & (y > 0)

    # exclude genes
    excl = np.array([str(g).lower() not in EXCLUDE_GENES for g in genes], dtype=bool)
    m = m & excl

    idx = np.where(m)[0]
    x_plot = x[idx]
    y_plot = y[idx]

    # per-class r / R²
    r_gene, r2_gene = pearson_r_and_r2_vec(pred_X, true_X)  # gene-wise (for highlight top_r2)
    # scatter-level r/R² uses plotted points
    r_scatter, r2_scatter = pearson_r_and_r2_vec(x_plot[None, :], y_plot[None, :])  # dummy shape
    # 上面 dummy 不合适，我们用标量计算：
    rr, rr2 = _pearson_scalar(x_plot, y_plot)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    ax.scatter(
        x_plot, y_plot, s=SCATTER_POINT_SIZE, alpha=SCATTER_POINT_ALPHA,
        c=point_color, edgecolors="none", label=str(cls_name)
    )

    # axis
    if AXIS_MODE == "fixed":
        lo, hi = AXIS_FIXED_LIM
        ax.plot([lo, hi], [lo, hi], "--", color=DIAG_LINE_COLOR, alpha=0.75, lw=1.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    else:
        if x_plot.size > 0:
            mx = float(max(x_plot.max(), y_plot.max()))
            mx = mx * (1.0 + AXIS_MARGIN_RATIO)
            mx = np.ceil(mx / SCATTER_TICK_STEP) * SCATTER_TICK_STEP
            ax.plot([0, mx], [0, mx], "--", color=DIAG_LINE_COLOR, alpha=0.75, lw=1.5)
            ax.set_xlim(0, mx); ax.set_ylim(0, mx)
            ax.xaxis.set_major_locator(MultipleLocator(SCATTER_TICK_STEP))
            ax.yaxis.set_major_locator(MultipleLocator(SCATTER_TICK_STEP))

    ax.set_aspect("equal", adjustable="box")

    # highlight（突出）
    if HIGHLIGHT_POINTS:
        hi = select_highlight_indices(cls_name, genes, x, y, r2_gene=r2_gene)
        if hi.size > 0:
            # only highlight genes that are actually plotted (avoid surprising)
            hi = np.array([g for g in hi if m[g]], dtype=int)

        if hi.size > 0:
            ax.scatter(
                x[hi], y[hi],
                s=HIGHLIGHT_POINT_SIZE,
                c=point_color,
                edgecolors=HIGHLIGHT_EDGE_COLOR,
                linewidth=HIGHLIGHT_EDGE_WIDTH,
                zorder=10
            )
            if HIGHLIGHT_LABELS:
                annotate_labels(ax, x[hi], y[hi], [genes[g] for g in hi])

    if np.isfinite(rr) and np.isfinite(rr2):
        ax.set_title(f"{title}\nPearson r={rr:.3f}, R²={rr2:.3f}", fontsize=13)
    else:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel("Pred mean")
    ax.set_ylabel("Real mean")
    ax.legend(frameon=False, loc="best")
    style_ax(ax, set_y_locator=False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)

    strip_plot_text_for_clean(ax, hide_tick_labels=True)
    fig.savefig(clean_variant_path(out_png), dpi=300)
    plt.close(fig)

    return rr, rr2, int(idx.size)

def _pearson_scalar(x, y, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return np.nan, np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    if denom < eps:
        return np.nan, np.nan
    r = float(np.sum(x*y) / denom)
    return r, float(r*r)

def _robust_ylim(values, qlo, qhi, margin_ratio):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo = np.percentile(v, qlo)
    hi = np.percentile(v, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < 1e-8:
        lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, 1e-6)
    pad = span * margin_ratio
    return float(lo - pad), float(hi + pad)

def plot_violin_per_class(pred_X, true_X, genes, out_png, title, cls_name):
    """
    violin: 每个基因两半（Real vs Pred）
    pred_X/true_X: (n, G) (cell-level 或 clone-level)
    """
    # gene-wise r2 for selection
    r, r2 = pearson_r_and_r2_vec(pred_X, true_X)

    n = true_X.shape[0]
    pos_counts = (true_X > 0).sum(axis=0).astype(np.int32)
    pos_fracs = pos_counts / max(n, 1)

    # select genes
    manual = VIOLIN_MANUAL_GENES_BY_CLASS.get(cls_name, [])
    manual = [g for g in manual if str(g).strip() != ""]
    name_to_idx = {str(g): i for i, g in enumerate(genes)}

    if VIOLIN_GENE_SELECT == "manual" and len(manual) > 0:
        sel = [name_to_idx[g] for g in manual if g in name_to_idx]
    else:
        cand = np.array([i for i in range(len(genes))
                         if (str(genes[i]).lower() not in EXCLUDE_GENES)
                         and (pos_counts[i] >= VIOLIN_MIN_POS_CELLS)
                         and (pos_fracs[i] >= VIOLIN_MIN_POS_FRAC)], dtype=int)
        if cand.size == 0:
            cand = np.array([i for i in range(len(genes)) if str(genes[i]).lower() not in EXCLUDE_GENES], dtype=int)

        score = np.nan_to_num(r2[cand], nan=-1e9)
        sel = cand[np.argsort(-score)[:VIOLIN_TOP_K]].tolist()

        # 如果 manual 也写了，想“manual优先 + 自动补足”，可这样：
        if len(manual) > 0:
            sel_manual = [name_to_idx[g] for g in manual if g in name_to_idx]
            sel_set = set(sel_manual)
            extra = [i for i in sel if i not in sel_set]
            sel = sel_manual + extra[:max(0, VIOLIN_TOP_K - len(sel_manual))]

    # build dataframe
    dfs = []
    for gi in sel:
        gname = str(genes[gi])
        t = true_X[:, gi].astype(np.float64)
        p = pred_X[:, gi].astype(np.float64)

        mask = np.isfinite(t) & np.isfinite(p)
        if VIOLIN_ONLY_EXPRESSED_TRUE:
            mask = mask & (t > 0.0)
            if int(mask.sum()) < 2:
                mask = np.isfinite(t) & np.isfinite(p)

        dfs.append(pd.DataFrame({"E": t[mask], "Type": "Real", "Gene": gname}))
        dfs.append(pd.DataFrame({"E": p[mask], "Type": "Pred", "Gene": gname}))

    df = pd.concat(dfs, ignore_index=True)

    # ylim
    yl = _robust_ylim(df["E"].values, VIOLIN_YLIM_PERCENTILES[0], VIOLIN_YLIM_PERCENTILES[1], VIOLIN_YLIM_MARGIN)
    if yl is not None:
        lo, hi = yl
        span = max(hi - lo, 1e-6)
        lo = lo - span * VIOLIN_BOTTOM_EXTRA_RATIO
        hi = hi + span * VIOLIN_TOP_HEADROOM_RATIO
        if VIOLIN_BOTTOM_FLOOR is not None:
            lo = min(lo, float(VIOLIN_BOTTOM_FLOOR))
        yl = (lo, hi)

    fig_w = max(10.0, 1.12 * len(sel) + 3.0)
    fig_h = 5.6
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    sns.violinplot(
        x="Gene", y="E", hue="Type", data=df,
        hue_order=["Real", "Pred"],
        palette=VIOLIN_PALETTE,
        inner=VIOLIN_INNER,
        linewidth=VIOLIN_LINEWIDTH,
        cut=VIOLIN_CUT,                # cut（切割）
        scale=VIOLIN_SCALE,
        bw_adjust=VIOLIN_BW_ADJUST,
        gridsize=VIOLIN_GRIDSIZE,
        ax=ax
    )

    if yl is not None:
        ax.set_ylim(yl[0], yl[1])

    # ✅ y 轴刻度：每 2 一个刻度
    ax.yaxis.set_major_locator(MultipleLocator(2))

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Expression")

    # ✅ 关键：不要让 style_ax 再用 MaxNLocator 覆盖你的 MultipleLocator
    style_ax(ax, set_y_locator=False)

    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h[:2], l[:2], frameon=True, title="")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    strip_plot_text_for_clean(ax, hide_tick_labels=True)
    plt.savefig(clean_variant_path(out_png), dpi=300)
    plt.close()

    # save gene table for this class
    tab = pd.DataFrame({
        "gene": [str(genes[i]) for i in sel],
        "r": [float(r[i]) if np.isfinite(r[i]) else np.nan for i in sel],
        "r2": [float(r2[i]) if np.isfinite(r2[i]) else np.nan for i in sel],
        "n_pos_true": [int(pos_counts[i]) for i in sel],
        "frac_pos_true": [float(pos_fracs[i]) for i in sel],
    })
    return sel, tab


# =========================================================
# ======================  MAIN  ============================
# =========================================================
def main():
    tasks = TASKS if TASKS is not None else list_tasks(REG_OUT_DIR)
    if not tasks:
        raise FileNotFoundError(f"No tasks found under: {REG_OUT_DIR}")
    print("[Tasks]", tasks)

    for task in tasks:
        tdir = os.path.join(REG_OUT_DIR, task)
        npz_path = os.path.join(tdir, NPZ_NAME)
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path, allow_pickle=True)

        k_pred = pick_key(data, ["pred_log", "pred", "y_pred"])
        k_true = pick_key(data, ["true_log", "true", "y_true"])
        k_lab  = pick_key(data, ["label", "cell_type", "y_label", "labels"])
        k_gene = pick_key(data, ["gene_names", "genes", "var_names"])
        k_clid = pick_key(data, ["clone_id", "clone", "clone_ids"])

        if k_pred is None or k_true is None or k_lab is None or k_gene is None:
            raise KeyError(f"[{task}] missing required keys. Have: {data.files}")

        pred = np.asarray(data[k_pred], dtype=np.float32)
        true = np.asarray(data[k_true], dtype=np.float32)
        label = decode_arr(data[k_lab]).astype(str)
        genes = decode_arr(data[k_gene]).astype(object)

        clone_id = decode_arr(data[k_clid]) if (k_clid is not None) else None

        out_dir = os.path.join(tdir, OUT_SUBDIR)
        ensure_dir(out_dir)

        classes = sorted(pd.unique(label).tolist())
        color_map = build_color_map(classes, CLASS_COLORS, palette_name=FALLBACK_PALETTE)
        print(f"\n[{task}] classes={classes}")
        print(f"[{task}] color_map={color_map}")

        scatter_rows = []

        for cls in classes:
            m = (label == cls)
            n_cls = int(m.sum())
            if n_cls < MIN_SAMPLES_PER_CLASS:
                print(f"[Skip][{task}][{cls}] n={n_cls} < {MIN_SAMPLES_PER_CLASS}")
                continue

            pred_sub = pred[m]
            true_sub = true[m]

            # scatter aggregation
            if (AGG_MODE_SCATTER == "clone") and (clone_id is not None):
                cid_sub = clone_id[m]
                pred_sc, _ = compute_clone_means(pred_sub, cid_sub)
                true_sc, _ = compute_clone_means(true_sub, cid_sub)
                agg_sc = "clone"
            else:
                pred_sc, true_sc = pred_sub, true_sub
                agg_sc = "cell"

            # violin aggregation
            if (AGG_MODE_VIOLIN == "clone") and (clone_id is not None):
                cid_sub = clone_id[m]
                pred_vi, _ = compute_clone_means(pred_sub, cid_sub)
                true_vi, _ = compute_clone_means(true_sub, cid_sub)
                agg_vi = "clone"
            else:
                pred_vi, true_vi = pred_sub, true_sub
                agg_vi = "cell"

            # scatter
            if PLOT_SCATTER:
                png = os.path.join(out_dir, f"scatter_{safe_name(cls)}.png")
                title = f"{task} | {cls} | n={n_cls} | agg={agg_sc}"
                rr, rr2, n_genes = plot_scatter_per_class(
                    pred_sc, true_sc, genes, png, title, color_map.get(cls, FALLBACK_COLOR), cls
                )
                scatter_rows.append({
                    "task": task,
                    "class": cls,
                    "n_samples": n_cls,
                    "agg_mode": agg_sc,
                    "n_genes_plotted": n_genes,
                    "pearson_r": rr,
                    "r2": rr2,
                    "scatter_png": os.path.basename(png),
                    "scatter_png_clean": os.path.basename(clean_variant_path(png)),
                    "color": color_map.get(cls, FALLBACK_COLOR),
                    "highlight_points": HIGHLIGHT_POINTS,
                    "highlight_labels": HIGHLIGHT_LABELS,
                    "highlight_selection": HIGHLIGHT_SELECTION,
                })

            # violin
            if PLOT_VIOLIN:
                pngv = os.path.join(out_dir, f"violin_{safe_name(cls)}.png")
                titlev = f"{task} | {cls} | n={n_cls} | agg={agg_vi}"
                sel, tab = plot_violin_per_class(pred_vi, true_vi, genes, pngv, titlev, cls)
                tab.to_csv(os.path.join(out_dir, f"violin_gene_table_{safe_name(cls)}.csv"), index=False)

        if PLOT_SCATTER:
            pd.DataFrame(scatter_rows).to_csv(os.path.join(out_dir, "scatter_r2_summary.csv"), index=False)
            print(f"[OK] {task}: saved scatter + scatter_r2_summary.csv under {out_dir}")
        if PLOT_VIOLIN:
            print(f"[OK] {task}: saved violin + violin_gene_table_<class>.csv under {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
