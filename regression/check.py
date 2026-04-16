import numpy as np, pandas as pd
npz = "/Users/wanghongye/python/scLineagetracer/regression/result/GSE99915/Reg_D28_from_D6_D9_D12_D15_D21/test_outputs.npz"
d = np.load(npz, allow_pickle=True)
print(pd.Series(d["label"].astype(str)).value_counts())
