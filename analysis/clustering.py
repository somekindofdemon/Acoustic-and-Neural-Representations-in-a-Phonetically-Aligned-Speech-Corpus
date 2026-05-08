"""
Section 9 – Hierarchical Clustering

9.1 Clustering of French oral vowels (Ward's linkage)
9.2 Consonants vs. vowels
9.3 Clustering of speakers (concatenated per-phoneme means)
9.4 Determining k (silhouette, dendrogram height, linguistic coherence)

Outputs
-------
results/figures/dendrograms.png
results/figures/silhouette_plots.png
results/tables/ari_scores.csv
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
import yaml

warnings.filterwarnings("ignore")

CONSONANTS = ['p', 'b', 't', 'd', 'k', 'ɡ', 'f', 'v', 's', 'z', 'ʃ', 'ʒ',
              'm', 'n', 'l', 'ʁ']

FRONT_BACK = {
    'i': 'front', 'e': 'front', 'ɛ': 'front', 'a': 'front',
    'y': 'front', 'ø': 'front', 'œ': 'front', 'ə': 'front',
    'u': 'back',  'o': 'back',  'ɔ': 'back',  'ɑ': 'back',
}

HIGH_MID_LOW = {
    'i': 'high', 'y': 'high', 'u': 'high',
    'e': 'mid',  'ø': 'mid',  'o': 'mid',
    'ɛ': 'mid',  'œ': 'mid',  'ɔ': 'mid',  'ə': 'mid',
    'a': 'low',  'ɑ': 'low',
}


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── Build centroid matrices ───────────────────────────────────────────────────

def acoustic_centroids(df: pd.DataFrame, phones: list) -> pd.DataFrame:
    df_v = df[df['phoneme'].isin(phones)].dropna(subset=['F1_lob', 'F2_lob'])
    c    = df_v.groupby('phoneme')[['F1_lob', 'F2_lob']].mean()
    return c.reindex(phones).dropna()


def neural_centroids(npz_path: str, phones: list, layer: int = None) -> tuple:
    data   = np.load(npz_path, allow_pickle=True)
    layers = data['layers'].tolist()
    L      = layer if (layer in layers) else layers[0]
    feats  = data[f'layer_{L}']
    ph_arr = data['phoneme'].astype(str)
    valid  = ~np.isnan(feats).any(axis=1)
    feats  = feats[valid]
    ph_arr = ph_arr[valid]

    rows = {}
    for ph in phones:
        mask = ph_arr == ph
        if mask.sum() > 0:
            rows[ph] = feats[mask].mean(axis=0)
    available = [p for p in phones if p in rows]
    if not available:
        return available, np.empty((0, feats.shape[1]))
    return available, np.stack([rows[p] for p in available])


# ── Ward's clustering ─────────────────────────────────────────────────────────

def ward_cluster(dist_vec: np.ndarray) -> np.ndarray:
    """Ward's hierarchical clustering from condensed distance vector."""
    return sch.linkage(dist_vec, method='ward')


# ── ARI evaluation ────────────────────────────────────────────────────────────

def eval_ari(Z: np.ndarray, labels_true: list, k: int) -> float:
    """Cut dendrogram at k clusters and compute ARI."""
    clusters = sch.fcluster(Z, k, criterion='maxclust')
    le       = LabelEncoder()
    gt       = le.fit_transform(labels_true)
    return adjusted_rand_score(gt, clusters)


# ── Silhouette scores ─────────────────────────────────────────────────────────

def silhouette_scan(X: np.ndarray, Z: np.ndarray,
                    k_range: range, metric: str) -> dict:
    scores = {}
    for k in k_range:
        if k >= len(X):
            break
        labels = sch.fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) < 2:
            continue
        try:
            scores[k] = silhouette_score(X, labels, metric=metric)
        except Exception:
            pass
    return scores


# ── Dendrogram plot ───────────────────────────────────────────────────────────

def plot_dendrogram(Z: np.ndarray, labels: list, title: str, ax,
                    color_threshold=None):
    sch.dendrogram(Z, labels=labels, ax=ax,
                   color_threshold=color_threshold,
                   leaf_rotation=90, leaf_font_size=9,
                   above_threshold_color='grey')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Distance')


# ── 9.3 Speaker clustering ────────────────────────────────────────────────────

def speaker_centroid_matrix(df_ac: pd.DataFrame, npz_w, npz_x,
                             vowels: list) -> dict:
    """Build per-speaker representation vectors (concat of per-phoneme means)."""
    speakers = sorted(df_ac['speaker_id'].unique())
    l1_map   = df_ac.groupby('speaker_id')['L1'].first().to_dict()
    gen_map  = df_ac.groupby('speaker_id')['gender'].first().to_dict()

    mats = {}

    # Acoustic
    df_v = df_ac[df_ac['phoneme'].isin(vowels)].dropna(subset=['F1_lob', 'F2_lob'])
    ac_rows = []
    for spk in speakers:
        sub = df_v[df_v['speaker_id'] == spk]
        vec = []
        for ph in vowels:
            ph_sub = sub[sub['phoneme'] == ph][['F1_lob', 'F2_lob']].dropna()
            vec.extend(ph_sub.mean().values if len(ph_sub) > 0 else [np.nan, np.nan])
        ac_rows.append(vec)
    X_ac = np.array(ac_rows)
    # Impute column-wise means
    col_means = np.nanmean(X_ac, axis=0)
    for j in range(X_ac.shape[1]):
        X_ac[np.isnan(X_ac[:, j]), j] = col_means[j]
    mats['acoustic'] = X_ac

    # Neural
    for rep_label, npz_path in [('whisper', npz_w), ('xlsr', npz_x)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data   = np.load(npz_path, allow_pickle=True)
        L      = data['layers'][0]
        feats  = data[f'layer_{L}']
        ph_arr = data['phoneme'].astype(str)
        spk_arr = data['speaker_id'].astype(str)
        valid  = ~np.isnan(feats).any(axis=1)
        feats  = feats[valid]
        ph_arr = ph_arr[valid]
        spk_arr = spk_arr[valid]

        nn_rows = []
        for spk in speakers:
            vec_parts = []
            for ph in vowels:
                mask = (spk_arr == spk) & (ph_arr == ph)
                if mask.sum() > 0:
                    vec_parts.append(feats[mask].mean(axis=0))
                else:
                    vec_parts.append(np.full(feats.shape[1], np.nan))
            vec = np.concatenate(vec_parts)
            nn_rows.append(vec)
        X_nn = np.array(nn_rows)
        col_m = np.nanmean(X_nn, axis=0)
        for j in range(X_nn.shape[1]):
            X_nn[np.isnan(X_nn[:, j]), j] = col_m[j]
        mats[rep_label] = X_nn

    return speakers, l1_map, gen_map, mats


# ── Main ─────────────────────────────────────────────────────────────────────

def acoustic_consonant_vowel_ari(df_ac: pd.DataFrame, vowels: list,
                                  min_tokens: int = 20) -> list:
    """
    Acoustic consonant-vs-vowel clustering using SCG + duration_ms.

    For vowels F1_lob/F2_lob are also available but SCG+duration span
    the full phoneme inventory.  All features are z-scored globally before
    centroid computation so scales are comparable.

    Returns list of result dicts compatible with consonant_vowel_ari().
    """
    cons = [c for c in CONSONANTS if (df_ac['phoneme'] == c).sum() >= min_tokens]
    if len(cons) < 6:
        return []

    feat_cols = [c for c in ['SCG', 'duration_ms'] if c in df_ac.columns]
    if not feat_cols:
        return []

    all_phones = vowels + cons
    df_ph = df_ac[df_ac['phoneme'].isin(all_phones)].copy()
    df_ph = df_ph.dropna(subset=feat_cols)

    X = df_ph[feat_cols].values.astype(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    X_z = (X - mu) / sd

    ph_arr = df_ph['phoneme'].values
    centroids = {}
    for ph in all_phones:
        mask = ph_arr == ph
        if mask.sum() > 0:
            centroids[ph] = X_z[mask].mean(axis=0)

    available = [p for p in all_phones if p in centroids]
    if len(available) < 4:
        return []

    C   = np.stack([centroids[p] for p in available])
    D   = pdist(C, metric='euclidean')
    Z   = ward_cluster(D)
    gt  = ['vowel' if p in vowels else 'consonant' for p in available]
    ari = eval_ari(Z, gt, 2)

    return [{
        'representation': f'acoustic_{"+".join(feat_cols)}',
        'n_vowels': sum(1 for p in available if p in vowels),
        'n_consonants': sum(1 for p in available if p not in vowels),
        'consonants_used': ','.join(p for p in available if p not in vowels),
        'features_used': '+'.join(feat_cols),
        'partition': 'consonant_vs_vowel',
        'k': 2, 'ARI': ari,
    }]


def consonant_vowel_ari(npz_path: str, vowels: list,
                         rep_label: str, min_tokens: int = 20) -> list:
    """
    Cluster consonants + vowels together using neural centroids.
    ARI measures whether Ward clustering recovers the consonant/vowel boundary.
    Returns list of result dicts.
    """
    data   = np.load(npz_path, allow_pickle=True)
    layers = data['layers'].tolist()
    L      = layers[0]
    feats  = data[f'layer_{L}']
    ph_arr = data['phoneme'].astype(str)
    valid  = ~np.isnan(feats).any(axis=1)
    feats  = feats[valid]
    ph_arr = ph_arr[valid]

    consonants = [ph for ph in CONSONANTS if (ph_arr == ph).sum() >= min_tokens]
    if len(consonants) < 6:
        return []

    all_phones = vowels + consonants
    centroids  = {}
    for ph in all_phones:
        mask = ph_arr == ph
        if mask.sum() > 0:
            centroids[ph] = feats[mask].mean(axis=0)

    available = [p for p in all_phones if p in centroids]
    if len(available) < 4:
        return []

    C      = np.stack([centroids[p] for p in available])
    D      = pdist(C, metric='cosine')
    Z      = ward_cluster(D)
    gt_bin = ['vowel' if p in vowels else 'consonant' for p in available]
    ari_2  = eval_ari(Z, gt_bin, 2)

    rows = [{
        'representation': f'{rep_label}_layer{L}',
        'n_vowels': sum(1 for p in available if p in vowels),
        'n_consonants': sum(1 for p in available if p not in vowels),
        'consonants_used': ','.join(p for p in available if p not in vowels),
        'features_used': f'layer_{L}_hidden_states',
        'partition': 'consonant_vs_vowel',
        'k': 2, 'ARI': ari_2,
    }]
    return rows


def main(acoustic_norm, whisper_pca, xlsr_pca,
         out_dendro, out_ari, out_cv_ari,
         french_vowels, seed):

    for p in [out_dendro, out_ari, out_cv_ari]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    df_ac = pd.read_csv(acoustic_norm)
    ari_rows = []

    # ── 9.1 Vowel clustering ──────────────────────────────────────────────────
    print("Vowel clustering …")
    ac_c = acoustic_centroids(df_ac, french_vowels)
    valid_vows = ac_c.index.tolist()
    n_vows     = len(valid_vows)

    if n_vows >= 3:
        D_ac_vow  = pdist(ac_c.values, metric='euclidean')
        Z_ac_vow  = ward_cluster(D_ac_vow)

        for k, partition_map in [(2, FRONT_BACK), (3, HIGH_MID_LOW)]:
            labels_gt = [partition_map.get(v, 'unknown') for v in valid_vows]
            ari = eval_ari(Z_ac_vow, labels_gt, k)
            ari_rows.append({'representation': 'acoustic', 'phoneme_set': 'vowels',
                             'partition': 'front_back' if k == 2 else 'high_mid_low',
                             'k': k, 'ARI': ari})

        # Silhouette
        sil = silhouette_scan(ac_c.values, Z_ac_vow,
                              range(2, min(n_vows, 8)), 'euclidean')
        print(f"  Acoustic silhouette: {sil}")

    # Neural vowel clustering
    for rep_label, npz_path in [('whisper', whisper_pca), ('xlsr', xlsr_pca)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        avail, C = neural_centroids(npz_path, french_vowels)
        if len(avail) < 3:
            continue
        D_nn = pdist(C, metric='cosine')
        Z_nn = ward_cluster(D_nn)

        for k, partition_map, part_name in [
                (2, FRONT_BACK, 'front_back'),
                (3, HIGH_MID_LOW, 'high_mid_low')]:
            labels_gt = [partition_map.get(v, 'unknown') for v in avail]
            ari = eval_ari(Z_nn, labels_gt, k)
            ari_rows.append({'representation': rep_label, 'phoneme_set': 'vowels',
                             'partition': part_name, 'k': k, 'ARI': ari})

    # ── 9.2 Consonants + vowels ───────────────────────────────────────────────
    print("Consonant vs vowel clustering …")
    cv_rows = []
    # Acoustic: SCG + duration_ms as shared feature space
    cv_rows.extend(acoustic_consonant_vowel_ari(df_ac, french_vowels))
    # Neural: raw layer features
    for rep_label, npz_path in [('whisper', whisper_pca), ('xlsr', xlsr_pca)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        cv_rows.extend(consonant_vowel_ari(npz_path, french_vowels, rep_label))
    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(out_cv_ari, index=False)
    print(f"  Saved {out_cv_ari} ({len(df_cv)} rows)")

    # ── 9.3 Speaker clustering ────────────────────────────────────────────────
    print("Speaker clustering …")
    speakers, l1_map, gen_map, spk_mats = speaker_centroid_matrix(
        df_ac, whisper_pca, xlsr_pca, french_vowels)

    l1_gt  = LabelEncoder().fit_transform([l1_map[s] for s in speakers])
    gen_gt = LabelEncoder().fit_transform([gen_map[s] for s in speakers])

    for rep_label, X in spk_mats.items():
        D  = pdist(X, metric='euclidean' if rep_label == 'acoustic' else 'cosine')
        Z  = ward_cluster(D)
        for k, gt, part in [(2, l1_gt, 'L1_status'), (2, gen_gt, 'gender')]:
            if k >= len(speakers):
                continue
            ari = eval_ari(Z, gt.tolist(), k)
            ari_rows.append({'representation': rep_label, 'phoneme_set': 'speakers',
                             'partition': part, 'k': k, 'ARI': ari})

    # ── Save ARI table ────────────────────────────────────────────────────────
    df_ari = pd.DataFrame(ari_rows)
    df_ari.to_csv(out_ari, index=False)
    print(f"  Saved {out_ari}")

    # ── Dendrogram figure ─────────────────────────────────────────────────────
    n_panels = 1
    for npz_path in [whisper_pca, xlsr_pca]:
        if npz_path and os.path.exists(npz_path):
            n_panels += 1
    n_panels += len(spk_mats)

    fig, axes = plt.subplots(2, max(n_panels, 2), figsize=(5 * n_panels, 10))

    # Row 0: vowel dendrograms
    col = 0
    if n_vows >= 3:
        plot_dendrogram(Z_ac_vow, valid_vows, 'Acoustic – vowels', axes[0, col])
        col += 1

    for rep_label, npz_path in [('Whisper', whisper_pca), ('XLS-R', xlsr_pca)]:
        if npz_path and os.path.exists(npz_path) and col < axes.shape[1]:
            avail, C = neural_centroids(npz_path, french_vowels)
            if len(avail) >= 3:
                D = pdist(C, metric='cosine')
                Z = ward_cluster(D)
                plot_dendrogram(Z, avail, f'{rep_label} – vowels', axes[0, col])
                col += 1

    # Row 1: speaker dendrograms
    col = 0
    for rep_label, X in spk_mats.items():
        if col >= axes.shape[1]:
            break
        D = pdist(X, metric='euclidean' if rep_label == 'acoustic' else 'cosine')
        Z = ward_cluster(D)
        ax = axes[1, col]
        plot_dendrogram(Z, speakers, f'{rep_label} – speakers', ax)
        # Annotate with L1
        for tick_label in ax.get_xticklabels():
            spk = tick_label.get_text()
            l1  = l1_map.get(spk, '?')
            g   = gen_map.get(spk, '?')
            tick_label.set_text(f"{spk}\n({l1},{g})")
            tick_label.set_fontsize(7)
        col += 1

    # Hide unused panels
    for r in range(2):
        for c in range(col, axes.shape[1]):
            axes[r, c].set_visible(False)

    fig.suptitle("Hierarchical clustering (Ward's linkage)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dendro, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_dendro}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        acoustic_norm = snakemake.input.acoustic_norm,
        whisper_pca   = snakemake.input.whisper_pca,
        xlsr_pca      = snakemake.input.xlsr_pca,
        out_dendro    = snakemake.output.dendrograms,
        out_ari       = snakemake.output.ari_scores,
        out_cv_ari    = snakemake.output.consonant_vowel_ari,
        french_vowels = list(snakemake.params.french_vowels),
        seed          = snakemake.params.seed,
    )
except NameError:
    cfg = _load_config()
    d, r = cfg['data_dir'], cfg['results_dir']
    main(
        acoustic_norm = f"{d}/features_acoustic_norm.csv",
        whisper_pca   = f"{d}/features_whisper_pca.npz",
        xlsr_pca      = f"{d}/features_xlsr_pca.npz",
        out_dendro    = f"{r}/figures/dendrograms.png",
        out_ari       = f"{r}/tables/ari_scores.csv",
        out_cv_ari    = f"{r}/tables/consonant_vowel_ari.csv",
        french_vowels = cfg['french_oral_vowels'],
        seed          = cfg['random_seed'],
    )
