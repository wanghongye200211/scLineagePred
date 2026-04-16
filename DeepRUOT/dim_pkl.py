from DeepRUOT.utils import load_and_merge_config, euler_sdeint
from DeepRUOT.models import FNet, scoreNet2
from DeepRUOT.constants import RES_DIR
from DeepRUOT.exp import setup_exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

from DeepRUOT.losses import OT_loss1
from DeepRUOT.utils import (
    generate_steps, load_and_merge_config,
    SchrodingerBridgeConditionalFlowMatcher,
    generate_state_trajectory, get_batch, get_batch_size
)
from DeepRUOT.train import train_un1_reduce, train_all
from DeepRUOT.models import FNet, scoreNet2
from DeepRUOT.constants import DATA_DIR, RES_DIR
from DeepRUOT.exp import setup_exp
# Load and merge configuration
config_path = "/Users/wanghongye/python/scLineagetracer/DeepRUOT/config/GSE132188.yaml"
DATA_PATH    = "/Users/wanghongye/python/scLineagetracer/GSE132188/preprocess_final/ruot_input_pca50_forward.csv"
config = load_and_merge_config(config_path)

df = pd.read_csv(os.path.join(DATA_DIR, config['data']['file_path']))

# df = pd.read_csv('D:\DeepRUOTv2\data\elegan_pca.csv')
df = df.iloc[:, :config['data']['dim'] + 1]
device = torch.device('cpu')
exp_dir, logger = setup_exp(
    RES_DIR,
    config,
    config['exp']['name']
)
dim = config['data']['dim']
model_config = config['model']

f_net = FNet(
    in_out_dim=model_config['in_out_dim'],
    hidden_dim=model_config['hidden_dim'],
    n_hiddens=model_config['n_hiddens'],
    activation=model_config['activation']
).to(device)

sf2m_score_model = scoreNet2(
    in_out_dim=model_config['in_out_dim'],
    hidden_dim=model_config['score_hidden_dim'],
    activation=model_config['activation']
).float().to(device)
f_net.load_state_dict(torch.load(os.path.join(exp_dir, 'model_final'), map_location=torch.device('cpu')))
f_net.to(device)
sf2m_score_model.load_state_dict(torch.load(os.path.join(exp_dir, 'score_model_final'), map_location=torch.device('cpu')))
sf2m_score_model.to(device)

import numpy as np
import joblib
from sklearn.decomposition import PCA  # Import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import umap
import umap
import joblib


umap_op = umap.UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.01,
    spread=0.8,
    metric="euclidean",
    random_state=42,
    transform_seed=42,
    repulsion_strength=1.2
)

xu = umap_op.fit_transform(df.iloc[:, 1:])
joblib.dump(umap_op, os.path.join(exp_dir, "dim_reduction.pkl"))
print("[OK] saved:", os.path.join(exp_dir, "dim_reduction.pkl"))
