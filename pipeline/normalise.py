"""
Stage 5 – normalise
• Lobanov-normalise F1/F2 (vowels only, per speaker).
• PCA on neural representations → d=50 (analysis) and d=2 (viz).
• UMAP on neural representations → d=2 (viz).

Outputs:
  data/features_acoustic_norm.csv
  data/features_whisper_pca.npz
  data/features_xlsr_pca.npz

Each *_pca.npz contains:
  token_index, phoneme, speaker_id, sentence_id, L1, gender
  pca_{d}_layer_{L}   : (N, d) float32   PCA coordinates
  umap_layer_{L}      : (N, 2) float32   UMAP coordinates
  pca_explained_var_layer_{L}: (d,)      explained variance ratios
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import umap
import yaml

FRENCH_VOWELS = set("aɑeɛiɔouøœəɑ̃ɛ̃ɔ̃")


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── Lobanov normalisation ─────────────────────────────────────────────────────

def lobanov(df: pd.DataFrame, formants=('F1', 'F2')) -> pd.DataFrame:
    """
    For each speaker, normalise each formant F by:
        F* = (F - mean(F_speaker)) / std(F_speaker)
    Computed only on vowel tokens; consonants get the same transform applied
    (so their z-scores are relative to the vowel distribution).
    """
    df = df.copy()
    vowel_mask = df['phoneme'].apply(lambda p: str(p).strip() in FRENCH_VOWELS)

    for spk, grp in df.groupby('speaker_id'):
        vow_grp = grp[grp['phoneme'].apply(lambda p: str(p).strip() in FRENCH_VOWELS)]
        for f in formants:
            vals = vow_grp[f].dropna()
            if len(vals) < 2:
                continue
            mu  = vals.mean()
            std = vals.std(ddof=1)
            if std == 0:
                continue
            df.loc[grp.index, f'{f}_lob'] = (grp[f] - mu) / std

    return df


# ── Neural normalisation ──────────────────────────────────────────────────────

def process_neural(npz_path: str, pca_n_analysis: int, pca_n_viz: int,
                   umap_neighbors: int, umap_min_dist: float,
                   seed: int, out_path: str):
    data = np.load(npz_path, allow_pickle=True)
    layers = data['layers'].tolist()

    save = {
        'token_index': data['token_index'],
        'layers':      data['layers'],
    }
    for key in ('phoneme', 'speaker_id', 'sentence_id', 'L1', 'gender'):
        if key in data:
            save[key] = data[key]

    for L in layers:
        key = f'layer_{L}'
        X   = data[key].astype(np.float32)

        # Impute NaNs (rows where extraction failed)
        imp = SimpleImputer(strategy='mean')
        X   = imp.fit_transform(X)

        # Save imputed raw features (needed by analysis scripts)
        save[f'layer_{L}'] = X

        # PCA – analysis dimensionality
        n_a  = min(pca_n_analysis, X.shape[1], X.shape[0] - 1)
        pca_a = PCA(n_components=n_a, random_state=seed)
        coords_a = pca_a.fit_transform(X).astype(np.float32)
        save[f'pca_{n_a}_layer_{L}']            = coords_a
        save[f'pca_explained_var_layer_{L}']    = pca_a.explained_variance_ratio_.astype(np.float32)

        # PCA – visualisation (d=2)
        pca_v = PCA(n_components=2, random_state=seed)
        save[f'pca_2_layer_{L}'] = pca_v.fit_transform(X).astype(np.float32)

        # UMAP – visualisation
        print(f"  UMAP layer {L} …", flush=True)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric='cosine',
            random_state=seed,
            verbose=False,
        )
        save[f'umap_layer_{L}'] = reducer.fit_transform(X).astype(np.float32)

    np.savez_compressed(out_path, **save)
    print(f"  Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(acoustic_in, whisper_in, xlsr_in,
         acoustic_out, whisper_out, xlsr_out,
         params):
    for p in (acoustic_out, whisper_out, xlsr_out):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # ── Acoustic: Lobanov ─────────────────────────────────────────────────────
    print("Lobanov normalising acoustics …")
    df = pd.read_csv(acoustic_in)
    df = lobanov(df, formants=['F1', 'F2'])
    df.to_csv(acoustic_out, index=False)
    print(f"  Wrote {acoustic_out}")

    # ── Neural ────────────────────────────────────────────────────────────────
    print("PCA + UMAP for Whisper …")
    process_neural(whisper_in,
                   params['pca_analysis'], params['pca_viz'],
                   params['umap_neighbors'], params['umap_min_dist'],
                   params['seed'], whisper_out)

    print("PCA + UMAP for XLS-R …")
    process_neural(xlsr_in,
                   params['pca_analysis'], params['pca_viz'],
                   params['umap_neighbors'], params['umap_min_dist'],
                   params['seed'], xlsr_out)


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    p = {
        'pca_analysis':   snakemake.params.pca_analysis,
        'pca_viz':        snakemake.params.pca_viz,
        'umap_neighbors': snakemake.params.umap_neighbors,
        'umap_min_dist':  snakemake.params.umap_min_dist,
        'seed':           snakemake.params.seed,
    }
    main(
        snakemake.input.acoustic, snakemake.input.whisper, snakemake.input.xlsr,
        snakemake.output.acoustic_norm, snakemake.output.whisper_pca, snakemake.output.xlsr_pca,
        p,
    )
except NameError:
    cfg = _load_config()
    p = {
        'pca_analysis':   cfg['pca_n_components_analysis'],
        'pca_viz':        cfg['pca_n_components_viz'],
        'umap_neighbors': cfg['umap_n_neighbors'],
        'umap_min_dist':  cfg['umap_min_dist'],
        'seed':           cfg['random_seed'],
    }
    d = cfg['data_dir']
    main(
        f"{d}/features_acoustic.csv",
        f"{d}/features_whisper.npz",
        f"{d}/features_xlsr.npz",
        f"{d}/features_acoustic_norm.csv",
        f"{d}/features_whisper_pca.npz",
        f"{d}/features_xlsr_pca.npz",
        p,
    )
