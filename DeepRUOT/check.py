import numpy as np, os
CONFIG_PATH = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/config/GSE175634.yaml"
EXP_DIR = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/results/GSE175634"

seq = np.load(os.path.join(EXP_DIR, "sde_seq_0.npy"), mmap_mode="r")
print(seq.shape)           # 应该是 (T, N0)
print(seq[0, :5])          # t0 应该是 df 的 t0 全局行号（identity）
print(np.unique(seq[-1]).size)  # 最后一个时间点匹配到的原始点数量（应当 > 1）
