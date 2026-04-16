# -*- coding: utf-8 -*-
import numpy as np
import h5py
from typing import Optional, Tuple

def _read_gene_names_txt(path: Optional[str], G: int):
    if path is None:
        return np.array([f"gene_{i}" for i in range(G)], dtype=object)
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                names.append(t)
    if len(names) != G:
        raise ValueError(f"基因名数量({len(names)})与矩阵列数({G})不一致")
    return np.array(names, dtype=object)

def load_h5_to_matrix(h5_path: str, gene_names_txt: Optional[str]=None, h5_key: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        if h5_key is not None:
            dset = f[h5_key][()]
        else:
            key = None
            def _walk(name, obj):
                nonlocal key
                if key is None and isinstance(obj, h5py.Dataset):
                    key = name
            f.visititems(_walk)
            if key is None:
                raise ValueError("未在H5中找到任何dataset")
            dset = f[key][()]
    X = np.asarray(dset).astype(np.float32)
    genes = _read_gene_names_txt(gene_names_txt, X.shape[1])
    return X, genes

def load_h5ad(h5ad_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import anndata as ad
    except Exception as e:
        raise ImportError("需要 anndata 才能读取 .h5ad") from e
    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.A
    genes = np.asarray(adata.var_names.values, dtype=object)
    return X.astype(np.float32), genes
