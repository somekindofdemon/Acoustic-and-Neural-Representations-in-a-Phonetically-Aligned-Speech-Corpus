"""
Section 5 – Descriptive Statistics

Outputs
-------
results/figures/vowel_chart.png         – F1×F2 centroids + 95% CI ellipses
results/figures/boxplots_f1_f2.png      – box plots per phoneme × group
results/figures/violin_intraspeaker.png – intra-speaker variability
results/figures/neural_projections.png  – PCA / UMAP projections (phoneme / L1 / gender)
results/tables/descriptive_stats.csv
results/tables/variance_decomposition.csv
results/tables/mantel_rsm.csv           – Mantel test (acoustic vs Whisper vs XLS-R RSMs)
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist
import yaml

warnings.filterwarnings("ignore")

GROUPS = {
    ('fr', 'f'): ('L1F',  '#1f77b4'),
    ('fr', 'm'): ('L1M',  '#aec7e8'),
    ('ru', 'f'): ('L2F',  '#d62728'),
    ('ru', 'm'): ('L2M',  '#ff9896'),
}


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── Confidence ellipse ────────────────────────────────────────────────────────

def confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    if len(x) < 3:
        return
    cov   = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order  = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h   = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell    = Ellipse(xy=(np.mean(x), np.mean(y)),
                     width=w, height=h, angle=theta,
                     linewidth=1.5, fill=False, **kwargs)
    ax.add_patch(ell)


# ── Vowel chart (Q1) ─────────────────────────────────────────────────────────

def plot_vowel_chart(df: pd.DataFrame, vowels: list, out: str):
    df_v = df[df['phoneme'].isin(vowels)].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    for (l1, g), (label, colour) in GROUPS.items():
        sub = df_v[(df_v['L1'] == l1) & (df_v['gender'] == g)]
        if sub.empty:
            continue
        for ph in vowels:
            ph_sub = sub[sub['phoneme'] == ph].dropna(subset=['F1_lob', 'F2_lob'])
            if len(ph_sub) < 3:
                continue
            cx = ph_sub['F2_lob'].mean()
            cy = ph_sub['F1_lob'].mean()
            ax.scatter(cx, cy, color=colour, s=18, zorder=3)
            confidence_ellipse(ph_sub['F2_lob'].values,
                                ph_sub['F1_lob'].values,
                                ax, n_std=1.96, edgecolor=colour, alpha=0.5)
            ax.text(cx + 0.02, cy + 0.02, ph, fontsize=7, color=colour)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Lobanov z-score, ← back  front →)')
    ax.set_ylabel('F1 (Lobanov z-score, ↑ high  low ↓)')
    ax.set_title('French oral vowel space by speaker group')
    handles = [mpatches.Patch(color=c, label=l) for (_, _), (l, c) in GROUPS.items()]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Box plots (Q2) ───────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, vowels: list, out: str):
    df_v = df[df['phoneme'].isin(vowels)].dropna(subset=['F1_lob', 'F2_lob'])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for ax, feat in zip(axes, ['F1_lob', 'F2_lob']):
        groups_data = {}
        for ph in vowels:
            for (l1, g), (label, _) in GROUPS.items():
                vals = df_v[(df_v['phoneme'] == ph) &
                            (df_v['L1'] == l1) & (df_v['gender'] == g)][feat].dropna()
                groups_data.setdefault(ph, {})[label] = vals.values

        positions = []
        tick_pos  = []
        labels_   = []
        box_data  = []
        colours   = []
        pos       = 1
        for ph in vowels:
            tick_pos.append(pos + 1.5)
            labels_.append(ph)
            for label, colour in [(l, c) for (_, _), (l, c) in GROUPS.items()]:
                vals = groups_data.get(ph, {}).get(label, np.array([]))
                box_data.append(vals)
                positions.append(pos)
                colours.append(colour)
                pos += 1
            pos += 1

        bp = ax.boxplot(box_data, positions=positions,
                        patch_artist=True, widths=0.7,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        whiskerprops={'linewidth': 1},
                        capprops={'linewidth': 1},
                        flierprops={'markersize': 2})
        for patch, c in zip(bp['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels_, fontsize=9)
        ax.set_ylabel(feat)
        ax.set_title(f'{feat} per phoneme by group')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')

    handles = [mpatches.Patch(color=c, label=l) for (_, _), (l, c) in GROUPS.items()]
    axes[0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Intra-speaker violin (Q3) ─────────────────────────────────────────────────

def plot_intraspeaker_violin(df: pd.DataFrame, out: str):
    subset_vowels = ['i', 'a', 'u', 'y']
    df_v = df[df['phoneme'].isin(subset_vowels)].dropna(subset=['F1_lob'])

    # Compute per-speaker SD per phoneme
    rows = []
    for (spk, ph), g in df_v.groupby(['speaker_id', 'phoneme']):
        if len(g) >= 3:
            rows.append({'speaker_id': spk, 'phoneme': ph,
                         'F1_std': g['F1_lob'].std(), 'L1': g['L1'].iloc[0]})
    df_sd = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(subset_vowels))
    for j, ph in enumerate(subset_vowels):
        for offset, l1_val, colour in [(-0.2, 'fr', '#1f77b4'), (0.2, 'ru', '#d62728')]:
            vals = df_sd[(df_sd['phoneme'] == ph) & (df_sd['L1'] == l1_val)]['F1_std'].values
            if len(vals) < 2:
                continue
            parts = ax.violinplot(vals, positions=[j + offset], widths=0.35,
                                  showmedians=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(colour)
                pc.set_alpha(0.6)
            parts['cmedians'].set_color('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(subset_vowels)
    ax.set_ylabel('Intra-speaker F1 SD (Lobanov)')
    ax.set_title('Intra-speaker F1 variability across repetitions')
    l1_patch = mpatches.Patch(color='#1f77b4', label='L1 (fr)', alpha=0.6)
    l2_patch = mpatches.Patch(color='#d62728', label='L2 (ru)', alpha=0.6)
    ax.legend(handles=[l1_patch, l2_patch])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Descriptive stats table ───────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame, vowels: list) -> pd.DataFrame:
    df_v = df[df['phoneme'].isin(vowels)]
    rows = []
    for ph in vowels:
        for (l1, g), (label, _) in GROUPS.items():
            sub = df_v[(df_v['phoneme'] == ph) &
                       (df_v['L1'] == l1) & (df_v['gender'] == g)]
            for feat in ['F1_lob', 'F2_lob']:
                vals = sub[feat].dropna()
                if len(vals) == 0:
                    continue
                rows.append({
                    'phoneme': ph, 'group': label, 'feature': feat,
                    'n': len(vals),
                    'mean': vals.mean(), 'median': vals.median(),
                    'sd': vals.std(), 'iqr': vals.quantile(0.75) - vals.quantile(0.25),
                    'cv': vals.std() / abs(vals.mean()) if vals.mean() != 0 else np.nan,
                })
    return pd.DataFrame(rows)


# ── Variance decomposition (Eq. 2) ───────────────────────────────────────────

def variance_decomposition(df: pd.DataFrame, vowels: list) -> pd.DataFrame:
    """
    One-way ANOVA decomposition: SS_total = SS_between + SS_within.
    SS_between = inter-speaker; SS_within = intra-speaker + residual.
    Percentages are guaranteed to sum to 100%.
    """
    rows = []
    df_v = df[df['phoneme'].isin(vowels)].dropna(subset=['F1_lob'])
    for ph in vowels:
        sub = df_v[df_v['phoneme'] == ph].copy()
        if len(sub) < 3:
            continue
        grand_mean = sub['F1_lob'].mean()
        N = len(sub)
        k = sub['speaker_id'].nunique()

        SS_total = ((sub['F1_lob'] - grand_mean) ** 2).sum()
        spk_stats = sub.groupby('speaker_id')['F1_lob'].agg(['count', 'mean'])
        SS_between = float(((spk_stats['count']) * (spk_stats['mean'] - grand_mean) ** 2).sum())
        SS_within  = SS_total - SS_between

        MS_between = SS_between / max(k - 1, 1)
        MS_within  = SS_within  / max(N - k, 1)
        MS_total   = SS_total   / max(N - 1, 1)

        rows.append({
            'phoneme': ph,
            'N': N, 'k_speakers': k,
            'var_total': MS_total,
            'var_inter_speaker': MS_between,
            'var_intra_residual': MS_within,
            'pct_inter': 100 * SS_between / SS_total if SS_total > 0 else np.nan,
            'pct_intra_residual': 100 * SS_within  / SS_total if SS_total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ── Neural projections ────────────────────────────────────────────────────────

def _load_npz_proj(npz_path):
    """Load PCA-2 and UMAP-2 projections plus metadata from a normalised NPZ."""
    data       = np.load(npz_path, allow_pickle=True)
    L          = data['layers'].tolist()[0]
    pca2_key   = f'pca_2_layer_{L}'
    umap_key   = f'umap_layer_{L}'
    if pca2_key not in data:
        return None
    pca2 = data[pca2_key]
    umap2 = data[umap_key] if umap_key in data else pca2
    return {
        'L': L,
        'pca2': pca2,
        'umap2': umap2,
        'phoneme':  data['phoneme'].astype(str),
        'L1':       data['L1'].astype(str),
        'gender':   data['gender'].astype(str),
    }


def _scatter_one(ax, proj, cat_arr, color_map: dict, title: str,
                 show_legend: bool = True):
    for cat, c in color_map.items():
        mask = cat_arr == cat
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=c, s=4, alpha=0.5, label=cat)
    if show_legend:
        ax.legend(fontsize=7, markerscale=3, loc='best',
                  framealpha=0.6, handlelength=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def plot_neural_projections_split(npz_whisper, npz_xlsr,
                                   out_phoneme: str,
                                   out_l1: str,
                                   out_gender: str):
    """Three separate 2×2 figures (Whisper/XLS-R rows × PCA/UMAP cols),
    one per coloring scheme (phoneme / L1 / gender).
    """
    sources = [('Whisper', npz_whisper), ('XLS-R', npz_xlsr)]
    projs   = []
    for label, path in sources:
        if path and os.path.exists(path):
            d = _load_npz_proj(path)
            if d is not None:
                d['label'] = label
                projs.append(d)

    # Build colour maps once (union of all phonemes)
    all_ph   = sorted({ph for d in projs for ph in d['phoneme']})
    ph_cmap  = plt.get_cmap('tab20', len(all_ph))
    ph_colors = {p: ph_cmap(i) for i, p in enumerate(all_ph)}
    l1_color  = {'fr': '#1f77b4', 'ru': '#d62728'}
    g_color   = {'f': '#e377c2',  'm': '#8c564b'}

    scheme_cfg = [
        ('phoneme', ph_colors,  out_phoneme, 'Coloured by phoneme label'),
        ('L1',      l1_color,   out_l1,      'Coloured by L1 status'),
        ('gender',  g_color,    out_gender,  'Coloured by gender'),
    ]

    for scheme, color_map, out_path, suptitle in scheme_cfg:
        n_rows = len(projs)
        if n_rows == 0:
            continue
        fig, axes = plt.subplots(n_rows, 2,
                                  figsize=(10, 5 * n_rows),
                                  squeeze=False)
        for row_i, d in enumerate(projs):
            cat_arr = d[scheme] if scheme in ('L1', 'gender') else d['phoneme']
            for col_i, (proj, embed) in enumerate([(d['pca2'], 'PCA'),
                                                    (d['umap2'], 'UMAP')]):
                show_leg = (scheme != 'phoneme')   # phoneme legend is too crowded
                _scatter_one(axes[row_i][col_i], proj, cat_arr, color_map,
                             title=f"{d['label']} {embed}",
                             show_legend=show_leg)

        # Phoneme-coloured plots get a compact legend in the first panel
        if scheme == 'phoneme' and projs:
            ax0 = axes[0][0]
            handles = [mpatches.Patch(color=ph_colors[p], label=p) for p in all_ph]
            ax0.legend(handles=handles, fontsize=7, ncol=3,
                       loc='upper right', framealpha=0.7, markerscale=1)

        fig.suptitle(f'Neural representation projections — {suptitle}', fontsize=12)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Between-class variance ratio ──────────────────────────────────────────────

def compute_neural_variance_ratios(npz_whisper, npz_xlsr,
                                    vowels: list) -> pd.DataFrame:
    """Ratio of between-phoneme variance to total variance in the 2D space.
    Computed for each model × DR method (PCA / UMAP)."""
    rows = []
    for rep_label, npz_path in [('whisper', npz_whisper), ('xlsr', npz_xlsr)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        d = _load_npz_proj(npz_path)
        if d is None:
            continue
        mask = np.isin(d['phoneme'], vowels)
        for embed, proj in [('PCA', d['pca2']), ('UMAP', d['umap2'])]:
            feats  = proj[mask]
            labels = d['phoneme'][mask]
            valid  = ~np.isnan(feats).any(axis=1)
            feats, labels = feats[valid], labels[valid]
            ratio  = between_class_variance_ratio(feats, labels)
            rows.append({'representation': rep_label, 'layer': d['L'],
                         'method': embed, 'between_class_var_ratio': round(ratio, 4)})
    return pd.DataFrame(rows)


# ── Within / between phoneme cosine similarity ────────────────────────────────

def compute_cosine_similarity_ratio(npz_whisper, npz_xlsr,
                                     vowels: list,
                                     n_sample: int = 4000,
                                     seed: int = 42) -> pd.DataFrame:
    """Mean cosine similarity within vs. between phoneme classes.

    Samples n_sample token pairs for both within and between to keep runtime
    tractable.  Uses raw layer vectors (not PCA) for maximum fidelity.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for rep_label, npz_path in [('whisper', npz_whisper), ('xlsr', npz_xlsr)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data   = np.load(npz_path, allow_pickle=True)
        L      = data['layers'].tolist()[0]
        feats  = data[f'layer_{L}']
        ph_arr = data['phoneme'].astype(str)
        mask   = np.isin(ph_arr, vowels) & ~np.isnan(feats).any(axis=1)
        feats  = feats[mask]
        ph_arr = ph_arr[mask]

        # L2-normalise rows once
        norms     = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats_n   = feats / norms

        # ── Within-phoneme pairs ──────────────────────────────────────────────
        within_sims = []
        per_ph = n_sample // max(len(vowels), 1)
        for ph in vowels:
            idx = np.where(ph_arr == ph)[0]
            if len(idx) < 2:
                continue
            k = min(per_ph, len(idx) * (len(idx) - 1) // 2)
            for _ in range(k):
                i, j = rng.choice(idx, 2, replace=False)
                within_sims.append(float(feats_n[i] @ feats_n[j]))

        # ── Between-phoneme pairs ─────────────────────────────────────────────
        between_sims = []
        ph_list = np.array(vowels)
        for _ in range(n_sample):
            ph1, ph2 = rng.choice(ph_list, 2, replace=False)
            idx1 = np.where(ph_arr == ph1)[0]
            idx2 = np.where(ph_arr == ph2)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            i = rng.choice(idx1)
            j = rng.choice(idx2)
            between_sims.append(float(feats_n[i] @ feats_n[j]))

        w_mean = float(np.mean(within_sims))  if within_sims  else np.nan
        b_mean = float(np.mean(between_sims)) if between_sims else np.nan
        ratio  = w_mean / b_mean if (b_mean and b_mean != 0) else np.nan

        rows.append({
            'representation':       rep_label,
            'layer':                L,
            'within_phoneme_sim':   round(w_mean, 4),
            'between_phoneme_sim':  round(b_mean, 4),
            'within_between_ratio': round(ratio, 4),
            'n_within_pairs':       len(within_sims),
            'n_between_pairs':      len(between_sims),
        })

    return pd.DataFrame(rows)


# ── RSM + Mantel test ─────────────────────────────────────────────────────────

def mantel_test(rsm1: np.ndarray, rsm2: np.ndarray,
                n_perm: int = 999, seed: int = 42) -> tuple:
    """Mantel test on upper triangles of two RSMs."""
    rng   = np.random.default_rng(seed)
    n     = rsm1.shape[0]
    idx   = np.triu_indices(n, k=1)
    v1    = rsm1[idx]
    v2    = rsm2[idx]
    obs_r = stats.spearmanr(v1, v2).statistic

    perm_r = np.empty(n_perm)
    perm_idx = np.arange(n)
    for k in range(n_perm):
        rng.shuffle(perm_idx)
        v2p       = rsm2[np.ix_(perm_idx, perm_idx)][idx]
        perm_r[k] = stats.spearmanr(v1, v2p).statistic

    p_val = (np.sum(np.abs(perm_r) >= np.abs(obs_r)) + 1) / (n_perm + 1)
    return float(obs_r), float(p_val)


def _build_phoneme_speaker_vectors(npz_path, vowels, use_pca: bool = False):
    """Per-(phoneme, speaker) mean vector from a neural NPZ.
    Returns (keys, X) where keys is list of (phoneme, speaker_id) tuples and
    X is the corresponding (N, d) feature matrix."""
    data    = np.load(npz_path, allow_pickle=True)
    L       = data['layers'].tolist()[0]
    if use_pca:
        pca_keys = sorted(
            [k for k in data.keys() if k.startswith('pca_') and
             k.endswith(f'_layer_{L}') and 'explained' not in k],
            key=lambda k: int(k.split('_')[1]), reverse=True)
        feats = data[pca_keys[0]] if pca_keys else data[f'layer_{L}']
    else:
        feats = data[f'layer_{L}']
    ph_arr  = data['phoneme'].astype(str)
    spk_arr = data['speaker_id'].astype(str)
    valid   = ~np.isnan(feats).any(axis=1) & np.isin(ph_arr, vowels)
    feats, ph_arr, spk_arr = feats[valid], ph_arr[valid], spk_arr[valid]

    keys, rows = [], []
    for ph in vowels:
        for spk in np.unique(spk_arr):
            mask = (ph_arr == ph) & (spk_arr == spk)
            if mask.sum() == 0:
                continue
            keys.append((ph, spk))
            rows.append(feats[mask].mean(axis=0))
    return keys, (np.array(rows) if rows else np.empty((0, feats.shape[1]))), L


def compute_rsms(df_ac: pd.DataFrame, npz_w, npz_x,
                 vowels: list, n_perm: int, seed: int) -> pd.DataFrame:
    """Cross-representation comparison via Mantel tests on per-(phoneme, speaker)
    Representational Similarity Matrices (Section 5.3).

    Each "entity" is one (phoneme, speaker) cell; the RSM is therefore
    ~190×190 (10 vowels × 19 speakers).  Acoustic similarity = -Euclidean,
    neural similarity = cosine.  Common entities are intersected pairwise.
    """
    rsm_dict = {}

    # ── Acoustic per-(phoneme, speaker) means in (F1_lob, F2_lob) space ──────
    df_v   = df_ac[df_ac['phoneme'].isin(vowels)].dropna(subset=['F1_lob', 'F2_lob'])
    ac_grp = df_v.groupby(['phoneme', 'speaker_id'])[['F1_lob', 'F2_lob']].mean()
    ac_keys = [(ph, spk) for (ph, spk) in ac_grp.index]
    ac_X    = ac_grp.values
    if len(ac_X) >= 5:
        rsm_dict['acoustic'] = (ac_keys, -cdist(ac_X, ac_X, 'euclidean'))

    # ── Neural per-(phoneme, speaker) means using full layer vectors ─────────
    for label, npz_path in [('whisper', npz_w), ('xlsr', npz_x)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        keys, X, L = _build_phoneme_speaker_vectors(npz_path, vowels)
        if len(X) < 5:
            continue
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn    = X / norms
        rsm_dict[label] = (keys, Xn @ Xn.T)   # cosine similarity

    # ── Pairwise Mantel tests on common (phoneme, speaker) keys ──────────────
    pairs = [('acoustic', 'whisper'),
             ('acoustic', 'xlsr'),
             ('whisper',  'xlsr')]
    rows = []
    for r1, r2 in pairs:
        if r1 not in rsm_dict or r2 not in rsm_dict:
            continue
        keys1, M1 = rsm_dict[r1]
        keys2, M2 = rsm_dict[r2]
        common = [k for k in keys1 if k in set(keys2)]
        if len(common) < 5:
            continue
        idx1 = [keys1.index(k) for k in common]
        idx2 = [keys2.index(k) for k in common]
        m1   = M1[np.ix_(idx1, idx1)]
        m2   = M2[np.ix_(idx2, idx2)]
        r, p = mantel_test(m1, m2, n_perm=n_perm, seed=seed)
        rows.append({
            'comparison': f'{r1} vs {r2}',
            'n_entities': len(common),
            'mantel_r':   round(r, 4),
            'p_value':    round(p, 4),
        })
    return pd.DataFrame(rows)


# ── Neural per-phoneme inter-speaker variance decomposition ──────────────────

def compute_neural_variance_decomposition(npz_w, npz_x,
                                           vowels: list) -> pd.DataFrame:
    """Multivariate ANOVA-style variance decomposition for neural representations.

    For each (phoneme, representation):
        SS_total   = Σ ||x_i - grand_mean||²
        SS_between = Σ_s n_s · ||x̄_s - grand_mean||²    (inter-speaker)
        SS_within  = SS_total − SS_between              (intra + residual)

    This parallels the acoustic `variance_decomposition` so Q2 can be answered
    by comparing pct_inter across representations.
    """
    rows = []
    for rep_label, npz_path in [('whisper', npz_w), ('xlsr', npz_x)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path, allow_pickle=True)
        L       = data['layers'].tolist()[0]
        feats   = data[f'layer_{L}']
        ph_arr  = data['phoneme'].astype(str)
        spk_arr = data['speaker_id'].astype(str)
        valid   = ~np.isnan(feats).any(axis=1)
        feats, ph_arr, spk_arr = feats[valid], ph_arr[valid], spk_arr[valid]

        for ph in vowels:
            mask = ph_arr == ph
            if mask.sum() < 10:
                continue
            X    = feats[mask]
            spks = spk_arr[mask]
            uniq = np.unique(spks)
            if len(uniq) < 2:
                continue

            grand    = X.mean(axis=0)
            ss_total = float(np.sum((X - grand) ** 2))
            ss_between = 0.0
            for s in uniq:
                X_s = X[spks == s]
                ss_between += len(X_s) * float(np.sum((X_s.mean(axis=0) - grand) ** 2))
            ss_within = ss_total - ss_between

            rows.append({
                'representation': rep_label, 'layer': L, 'phoneme': ph,
                'N': int(len(X)), 'k_speakers': int(len(uniq)),
                'pct_inter':         round(100 * ss_between / ss_total, 2)
                                       if ss_total > 0 else np.nan,
                'pct_intra_residual': round(100 * ss_within  / ss_total, 2)
                                       if ss_total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ── Between-class variance ratio ──────────────────────────────────────────────

def between_class_variance_ratio(feats: np.ndarray, labels: np.ndarray) -> float:
    """Ratio of between-class variance to total variance."""
    total_var = feats.var(axis=0).sum()
    grand_mean = feats.mean(axis=0)
    classes   = np.unique(labels)
    bc_var = sum(
        (feats[labels == c].shape[0] *
         np.sum((feats[labels == c].mean(axis=0) - grand_mean) ** 2))
        for c in classes
    ) / len(feats)
    return float(bc_var / total_var) if total_var > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main(acoustic_norm, whisper_pca, xlsr_pca,
         out_vowel_chart, out_boxplots, out_violin,
         out_proj_phoneme, out_proj_l1, out_proj_gender,
         out_desc_stats, out_var_decomp, out_mantel,
         out_neural_var_ratio, out_neural_sim_ratio,
         out_neural_var_decomp,
         french_vowels, seed):

    for p in [out_vowel_chart, out_boxplots, out_violin,
              out_proj_phoneme, out_proj_l1, out_proj_gender]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    for p in [out_desc_stats, out_var_decomp, out_mantel,
              out_neural_var_ratio, out_neural_sim_ratio,
              out_neural_var_decomp]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    df = pd.read_csv(acoustic_norm)

    print("Vowel chart …")
    plot_vowel_chart(df, french_vowels, out_vowel_chart)

    print("Box plots …")
    plot_boxplots(df, french_vowels, out_boxplots)

    print("Violin plots …")
    plot_intraspeaker_violin(df, out_violin)

    print("Descriptive stats table …")
    desc = compute_descriptive_stats(df, french_vowels)
    desc.to_csv(out_desc_stats, index=False)
    print(f"  Saved {out_desc_stats}")

    print("Variance decomposition …")
    vd = variance_decomposition(df, french_vowels)
    vd.to_csv(out_var_decomp, index=False)
    print(f"  Saved {out_var_decomp}")

    print("Neural projections (3 separate figures) …")
    plot_neural_projections_split(whisper_pca, xlsr_pca,
                                   out_proj_phoneme, out_proj_l1, out_proj_gender)

    print("Neural between-class variance ratio …")
    df_vr = compute_neural_variance_ratios(whisper_pca, xlsr_pca, french_vowels)
    df_vr.to_csv(out_neural_var_ratio, index=False)
    print(f"  Saved {out_neural_var_ratio}")

    print("Neural within/between cosine similarity …")
    df_sr = compute_cosine_similarity_ratio(whisper_pca, xlsr_pca,
                                             french_vowels, seed=seed)
    df_sr.to_csv(out_neural_sim_ratio, index=False)
    print(f"  Saved {out_neural_sim_ratio}")

    print("Neural inter-speaker variance decomposition …")
    nvd = compute_neural_variance_decomposition(whisper_pca, xlsr_pca, french_vowels)
    nvd.to_csv(out_neural_var_decomp, index=False)
    print(f"  Saved {out_neural_var_decomp}")

    print("RSM + Mantel tests (phoneme × speaker) …")
    mantel_df = compute_rsms(df, whisper_pca, xlsr_pca,
                             french_vowels, n_perm=999, seed=seed)
    mantel_df.to_csv(out_mantel, index=False)
    print(f"  Saved {out_mantel}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        acoustic_norm        = snakemake.input.acoustic_norm,
        whisper_pca          = snakemake.input.whisper_pca,
        xlsr_pca             = snakemake.input.xlsr_pca,
        out_vowel_chart      = snakemake.output.vowel_chart,
        out_boxplots         = snakemake.output.boxplots_f1_f2,
        out_violin           = snakemake.output.violin_intraspeaker,
        out_proj_phoneme     = snakemake.output.neural_proj_phoneme,
        out_proj_l1          = snakemake.output.neural_proj_l1,
        out_proj_gender      = snakemake.output.neural_proj_gender,
        out_desc_stats       = snakemake.output.descriptive_stats,
        out_var_decomp       = snakemake.output.variance_decomp,
        out_mantel           = snakemake.output.mantel_rsm,
        out_neural_var_ratio = snakemake.output.neural_variance_ratio,
        out_neural_sim_ratio = snakemake.output.neural_similarity_ratio,
        out_neural_var_decomp = snakemake.output.neural_variance_decomp,
        french_vowels        = list(snakemake.params.french_vowels),
        seed                 = snakemake.params.seed,
    )
except NameError:
    cfg = _load_config()
    d, r = cfg['data_dir'], cfg['results_dir']
    main(
        acoustic_norm        = f"{d}/features_acoustic_norm.csv",
        whisper_pca          = f"{d}/features_whisper_pca.npz",
        xlsr_pca             = f"{d}/features_xlsr_pca.npz",
        out_vowel_chart      = f"{r}/figures/vowel_chart.png",
        out_boxplots         = f"{r}/figures/boxplots_f1_f2.png",
        out_violin           = f"{r}/figures/violin_intraspeaker.png",
        out_proj_phoneme     = f"{r}/figures/neural_proj_phoneme.png",
        out_proj_l1          = f"{r}/figures/neural_proj_L1.png",
        out_proj_gender      = f"{r}/figures/neural_proj_gender.png",
        out_desc_stats       = f"{r}/tables/descriptive_stats.csv",
        out_var_decomp       = f"{r}/tables/variance_decomposition.csv",
        out_mantel           = f"{r}/tables/mantel_rsm.csv",
        out_neural_var_ratio = f"{r}/tables/neural_variance_ratio.csv",
        out_neural_sim_ratio = f"{r}/tables/neural_similarity_ratio.csv",
        out_neural_var_decomp = f"{r}/tables/neural_variance_decomposition.csv",
        french_vowels        = cfg['french_oral_vowels'],
        seed                 = cfg['random_seed'],
    )
