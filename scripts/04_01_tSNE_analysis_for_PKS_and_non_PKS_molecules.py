from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tsnecuda import TSNE

data_dir = Path('../data')
splits = ['train', 'val', 'test']

def find_split(split):
    candidates = [
        data_dir / split / f'baseline_{split}_ecfp4.parquet',
        data_dir / split / f'baseline_{split}_ecfp4.csv',
        data_dir / split / f'baseline_{split}.parquet',
        data_dir / split / f'baseline_{split}.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

paths = {s: find_split(s) for s in splits}


# Load and combine
dfs = []
for s, p in paths.items():
    if p is None:
        print(f'Warning: missing split {s}; skipping')
        continue
    if p.suffix == '.parquet':
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df['split'] = s
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Prepare features and labels (no subsampling)
fp_cols = [c for c in df_all.columns if str(c).startswith('fp_')]
fp_cols = sorted(fp_cols, key=lambda s: int(str(s).split('_')[1]))
X = df_all[fp_cols].to_numpy(dtype=np.float32)
y = (df_all['source'].astype(str) == 'PKS').astype(int).to_numpy()
df_plot = df_all.copy().reset_index(drop=True)


