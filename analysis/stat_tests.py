"""
Section 6 – Statistical Tests

6.1 Group comparisons (L1 vs L2, gender), with BH-FDR correction
6.2 Inter-phoneme distance matrices + Mantel test
    Nearest-centroid classifier (leave-one-speaker-out)

Outputs
-------
results/tables/group_comparisons.csv
results/tables/distance_matrices_mantel.csv
results/tables/classifier_results.csv
results/figures/confusion_matrices.png
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.preprocessing import LabelEncoder
import yaml

warnings.filterwarnings("ignore")


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── BH FDR correction ─────────────────────────────────────────────────────────

def bh_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple:
    """Benjamini-Hochberg FDR correction. Returns (rejected, p_adjusted)."""
    n = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    rejected_sorted = p_sorted <= thresholds
    # All hypotheses below the last rejected one are also rejected
    if rejected_sorted.any():
        last = np.where(rejected_sorted)[0][-1]
        rejected_sorted[:last + 1] = True
    p_adj = np.minimum(1, p_sorted * n / np.arange(1, n + 1))
    # monotonise
    for i in range(n - 2, -1, -1):
        p_adj[i] = min(p_adj[i], p_adj[i + 1])
    out_rejected = np.empty(n, dtype=bool)
    out_padj     = np.empty(n)
    out_rejected[order] = rejected_sorted
    out_padj[order]     = p_adj
    return out_rejected, out_padj


# ── 6.1 L1 vs L2 acoustic (Q5a) ──────────────────────────────────────────────

def test_l1_l2_acoustic(df: pd.DataFrame, vowels: list) -> pd.DataFrame:
    rows = []
    for ph in vowels:
        for feat in ['F1_lob', 'F2_lob']:
            sub = df[df['phoneme'] == ph].dropna(subset=[feat])
            l1  = sub[sub['L1'] == 'fr'][feat].values
            l2  = sub[sub['L1'] == 'ru'][feat].values
            if len(l1) < 3 or len(l2) < 3:
                continue

            _, p_sw_l1 = stats.shapiro(l1[:50]) if len(l1) > 50 else stats.shapiro(l1)
            _, p_sw_l2 = stats.shapiro(l2[:50]) if len(l2) > 50 else stats.shapiro(l2)
            _, p_lev   = stats.levene(l1, l2)
            normal = (p_sw_l1 > 0.05) and (p_sw_l2 > 0.05)

            if normal and p_lev > 0.05:
                stat, p_val = stats.ttest_ind(l1, l2, equal_var=True)
                test_name = 't-test'
            elif normal:
                stat, p_val = stats.ttest_ind(l1, l2, equal_var=False)
                test_name = 'Welch'
            else:
                stat, p_val = stats.mannwhitneyu(l1, l2, alternative='two-sided')
                test_name = 'Mann-Whitney'

            d = (np.mean(l1) - np.mean(l2)) / np.sqrt(
                (np.std(l1, ddof=1) ** 2 + np.std(l2, ddof=1) ** 2) / 2)

            rows.append({
                'phoneme': ph, 'feature': feat,
                'n_l1': len(l1), 'n_l2': len(l2),
                'mean_l1': np.mean(l1), 'mean_l2': np.mean(l2),
                'diff': np.mean(l1) - np.mean(l2),
                'cohens_d': d,
                'test': test_name,
                'stat': stat, 'p_raw': p_val,
                'p_sw_l1': p_sw_l1, 'p_sw_l2': p_sw_l2, 'p_levene': p_lev,
            })

    df_out = pd.DataFrame(rows)
    if len(df_out):
        _, p_adj = bh_correction(df_out['p_raw'].values)
        df_out['p_adj_BH'] = p_adj
        df_out['significant_BH'] = df_out['p_adj_BH'] < 0.05
    return df_out


# ── 6.1 L1 vs L2 neural (permutation test, Q5b) ──────────────────────────────

def permutation_cosine(grp1: np.ndarray, grp2: np.ndarray,
                       n_perm: int, seed: int) -> tuple:
    """Permutation test on cosine distance between group centroids."""
    rng   = np.random.default_rng(seed)
    c1    = grp1.mean(axis=0)
    c2    = grp2.mean(axis=0)
    obs   = float(cdist(c1[None], c2[None], metric='cosine')[0, 0])

    all_data = np.vstack([grp1, grp2])
    n1 = len(grp1)
    perm_dists = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(len(all_data))
        pc1  = all_data[perm[:n1]].mean(axis=0)
        pc2  = all_data[perm[n1:]].mean(axis=0)
        perm_dists[k] = float(cdist(pc1[None], pc2[None], metric='cosine')[0, 0])

    p_val = (np.sum(perm_dists >= obs) + 1) / (n_perm + 1)
    return obs, p_val


def test_l1_l2_neural(npz_path: str, vowels: list,
                      layer: int, n_perm: int, seed: int) -> pd.DataFrame:
    if npz_path is None or not os.path.exists(npz_path):
        return pd.DataFrame()
    data      = np.load(npz_path, allow_pickle=True)
    layers    = data['layers'].tolist()
    L         = layer if layer in layers else layers[0]
    feats     = data[f'layer_{L}']
    phonemes  = data['phoneme'].astype(str)
    l1_arr    = data['L1'].astype(str)

    rows = []
    for ph in vowels:
        mask = phonemes == ph
        grp_l1 = feats[mask & (l1_arr == 'fr')]
        grp_l2 = feats[mask & (l1_arr == 'ru')]
        if len(grp_l1) < 3 or len(grp_l2) < 3:
            continue
        # Drop NaN rows
        grp_l1 = grp_l1[~np.isnan(grp_l1).any(axis=1)]
        grp_l2 = grp_l2[~np.isnan(grp_l2).any(axis=1)]
        if len(grp_l1) < 3 or len(grp_l2) < 3:
            continue
        obs, p = permutation_cosine(grp_l1, grp_l2, n_perm, seed)
        rows.append({'phoneme': ph, 'layer': L, 'cosine_dist': obs, 'p_raw': p})

    df_out = pd.DataFrame(rows)
    if len(df_out):
        _, p_adj = bh_correction(df_out['p_raw'].values)
        df_out['p_adj_BH'] = p_adj
        df_out['significant_BH'] = df_out['p_adj_BH'] < 0.05
    return df_out


# ── 6.2 Distance matrices ─────────────────────────────────────────────────────

def mantel_test(v1: np.ndarray, v2: np.ndarray,
                n_perm: int = 999, seed: int = 42) -> tuple:
    rng   = np.random.default_rng(seed)
    n     = int(np.round((1 + np.sqrt(1 + 8 * len(v1))) / 2))
    # v1, v2 are already upper-triangle vectors
    obs_r = stats.spearmanr(v1, v2).statistic
    # Permute via index
    idx = np.arange(n)
    perm_r = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(idx)
        mat_perm = _reconstruct_sym(v2, n)[np.ix_(idx, idx)]
        perm_r[k] = stats.spearmanr(v1, mat_perm[np.triu_indices(n, k=1)]).statistic
    p = (np.sum(np.abs(perm_r) >= np.abs(obs_r)) + 1) / (n_perm + 1)
    return float(obs_r), float(p)


def _reconstruct_sym(v: np.ndarray, n: int) -> np.ndarray:
    mat = np.zeros((n, n))
    mat[np.triu_indices(n, k=1)] = v
    mat += mat.T
    return mat


def distance_matrices_mantel(df_ac: pd.DataFrame, npz_w, npz_x,
                              vowels: list, n_perm: int, seed: int) -> pd.DataFrame:
    df_v = df_ac[df_ac['phoneme'].isin(vowels)].dropna(subset=['F1_lob', 'F2_lob'])
    ac_c = df_v.groupby('phoneme')[['F1_lob', 'F2_lob']].mean()
    ac_c = ac_c.reindex(vowels).dropna()
    valid = ac_c.index.tolist()
    n    = len(valid)
    if n < 2:
        return pd.DataFrame()

    D_eu  = cdist(ac_c.values, ac_c.values, 'euclidean')
    cov   = np.cov(df_v[['F1_lob', 'F2_lob']].dropna().values.T)
    if np.linalg.matrix_rank(cov) < 2:
        cov = np.eye(2)
    cov_inv = np.linalg.pinv(cov)
    D_ma  = np.array([[mahalanobis(ac_c.values[i], ac_c.values[j], cov_inv)
                        for j in range(n)] for i in range(n)])

    triu   = np.triu_indices(n, k=1)
    v_eu   = D_eu[triu]
    v_ma   = D_ma[triu]

    rows = []
    neural_cos = {}      # cache cosine distance matrices for wh-vs-xl Mantel
    for label, npz_path in [('whisper', npz_w), ('xlsr', npz_x)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data   = np.load(npz_path, allow_pickle=True)
        L      = data['layers'][0]
        ph_arr = data['phoneme'].astype(str)
        key = f'layer_{L}'
        if key not in data:
            continue
        feats  = data[key]
        mask_valid = ~np.isnan(feats).any(axis=1)
        feats  = feats[mask_valid]
        ph_arr = ph_arr[mask_valid]

        centroids = {v: feats[ph_arr == v].mean(axis=0)
                     for v in valid if (ph_arr == v).sum() > 0}
        if len(centroids) < n:
            continue
        C = np.stack([centroids[v] for v in valid])
        D_cos = cdist(C, C, 'cosine')
        v_cos = D_cos[triu]
        neural_cos[label] = v_cos

        r_eu, p_eu = mantel_test(v_eu, v_cos, n_perm, seed)
        r_ma, p_ma = mantel_test(v_ma, v_cos, n_perm, seed)
        rows.append({'comparison': f'acoustic_euclidean vs {label}',
                     'mantel_r': round(r_eu, 4), 'p_value': round(p_eu, 4)})
        rows.append({'comparison': f'acoustic_mahalanobis vs {label}',
                     'mantel_r': round(r_ma, 4), 'p_value': round(p_ma, 4)})

    # Whisper vs XLS-R (cosine distance matrices)
    if 'whisper' in neural_cos and 'xlsr' in neural_cos:
        r_wx, p_wx = mantel_test(neural_cos['whisper'], neural_cos['xlsr'],
                                  n_perm, seed)
        rows.append({'comparison': 'whisper vs xlsr',
                     'mantel_r': round(r_wx, 4), 'p_value': round(p_wx, 4)})

    return pd.DataFrame(rows)


# ── 6.1 Gender residual test (paired across phonemes) ────────────────────────

def gender_residual_test(df: pd.DataFrame, vowels: list) -> pd.DataFrame:
    """After Lobanov normalisation, test whether a residual gender effect persists.

    For each phoneme we average F1_lob (and F2_lob) over speakers within each
    gender, yielding one (F-mean, M-mean) pair per phoneme.  We then run a
    paired test across phonemes (n = number of vowels with both genders
    represented).  Both paired t-test and Wilcoxon signed-rank are reported.
    """
    from scipy.stats import ttest_rel, wilcoxon
    rows = []
    for feat in ['F1_lob', 'F2_lob']:
        sub = df[df['phoneme'].isin(vowels)].dropna(subset=[feat])
        # speaker × phoneme means
        spk_ph = sub.groupby(['phoneme', 'speaker_id', 'gender'])[feat].mean().reset_index()
        # per (phoneme, gender) means
        ph_gen = (spk_ph.groupby(['phoneme', 'gender'])[feat].mean()
                        .unstack('gender').dropna())
        if len(ph_gen) < 3 or 'f' not in ph_gen.columns or 'm' not in ph_gen.columns:
            continue
        f_vals = ph_gen['f'].values
        m_vals = ph_gen['m'].values
        t_stat, t_p = ttest_rel(f_vals, m_vals)
        w_stat, w_p = wilcoxon(f_vals, m_vals)
        rows.append({
            'feature':       feat,
            'n_phonemes':    int(len(ph_gen)),
            'mean_F':        round(float(f_vals.mean()), 4),
            'mean_M':        round(float(m_vals.mean()), 4),
            'mean_diff_F-M': round(float(f_vals.mean() - m_vals.mean()), 4),
            'paired_t_stat': round(float(t_stat), 4),
            'paired_t_p':    round(float(t_p), 4),
            'wilcoxon_W':    round(float(w_stat), 4),
            'wilcoxon_p':    round(float(w_p), 4),
        })
    return pd.DataFrame(rows)


# ── Nearest-centroid classifier (leave-one-speaker-out) ──────────────────────

def nearest_centroid_loso(feats: np.ndarray, labels: np.ndarray,
                          speakers: np.ndarray) -> dict:
    """Leave-one-speaker-out nearest-centroid classification."""
    unique_spk = np.unique(speakers)
    all_true   = []
    all_pred   = []

    for spk in unique_spk:
        test_mask  = speakers == spk
        train_mask = ~test_mask
        X_train = feats[train_mask]
        y_train = labels[train_mask]
        X_test  = feats[test_mask]
        y_test  = labels[test_mask]

        # Compute centroids from training set
        unique_cls = np.unique(y_train)
        centroids  = np.stack([X_train[y_train == c].mean(axis=0)
                                for c in unique_cls])
        D_test  = cdist(X_test, centroids, metric='cosine')
        pred_idx = D_test.argmin(axis=1)
        y_pred   = unique_cls[pred_idx]

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc  = (all_true == all_pred).mean()
    classes = np.unique(all_true)

    # Per-class F1
    f1_per_class = {}
    for c in classes:
        tp = ((all_true == c) & (all_pred == c)).sum()
        fp = ((all_true != c) & (all_pred == c)).sum()
        fn = ((all_true == c) & (all_pred != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_per_class[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # Confusion matrix
    n_cls = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    conf   = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(all_true, all_pred):
        if t in cls_idx and p in cls_idx:
            conf[cls_idx[t], cls_idx[p]] += 1

    return {'accuracy': acc, 'f1_per_class': f1_per_class,
            'confusion': conf, 'classes': classes,
            'all_true': all_true, 'all_pred': all_pred}


def plot_confusion_matrix(conf: np.ndarray, classes: list, title: str, ax):
    im = ax.imshow(conf, cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(conf[i, j]),
                    ha='center', va='center', fontsize=5,
                    color='white' if conf[i, j] > conf.max() * 0.5 else 'black')
    return im


# ── McNemar test ─────────────────────────────────────────────────────────────

def mcnemar_test(true1: np.ndarray, pred1: np.ndarray,
                  true2: np.ndarray, pred2: np.ndarray) -> dict:
    """McNemar chi-square (with continuity correction) comparing two classifiers.
    Assumes true1 == true2 (same test instances in same order)."""
    from scipy.stats import chi2
    correct1 = (true1 == pred1)
    correct2 = (true2 == pred2)
    b = int(( correct1 & ~correct2).sum())
    c = int((~correct1 &  correct2).sum())
    if b + c == 0:
        return {'b': 0, 'c': 0, 'statistic': 0.0, 'p_value': 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = float(chi2.sf(stat, df=1))
    return {'b': b, 'c': c, 'statistic': float(stat), 'p_value': p}


# ── Pairwise phoneme distance bootstrap CIs ──────────────────────────────────

PHONEME_PAIRS = [('e', 'ɛ'), ('o', 'u'), ('y', 'u'), ('a', 'ɑ')]


def pairwise_distance_bootstrap_ci(feats: np.ndarray, ph_arr: np.ndarray,
                                    spk_arr: np.ndarray,
                                    pairs: list, metric: str,
                                    n_boot: int, seed: int) -> list:
    """Bootstrap 95% CI on pairwise phoneme-centroid distance.

    Speaker-level resampling: speakers are drawn with replacement so that
    within-speaker correlations are respected and CIs are not artificially narrow.

    Parameters
    ----------
    metric : 'cosine' for neural features, 'euclidean' for Lobanov z-scores.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    def centroid_dist(f1: np.ndarray, f2: np.ndarray) -> float:
        c1 = f1.mean(axis=0)
        c2 = f2.mean(axis=0)
        if metric == 'cosine':
            return float(cdist(c1[None], c2[None], 'cosine')[0, 0])
        return float(np.linalg.norm(c1 - c2))

    for ph1, ph2 in pairs:
        m1 = ph_arr == ph1
        m2 = ph_arr == ph2
        if m1.sum() < 2 or m2.sum() < 2:
            continue
        f1, s1 = feats[m1], spk_arr[m1]
        f2, s2 = feats[m2], spk_arr[m2]
        spks1  = np.unique(s1)
        spks2  = np.unique(s2)
        if len(spks1) < 2 or len(spks2) < 2:
            continue

        obs  = centroid_dist(f1, f2)
        boot = np.empty(n_boot)
        for k in range(n_boot):
            bs1 = rng.choice(spks1, len(spks1), replace=True)
            bs2 = rng.choice(spks2, len(spks2), replace=True)
            bf1 = np.vstack([f1[s1 == s] for s in bs1 if (s1 == s).any()])
            bf2 = np.vstack([f2[s2 == s] for s in bs2 if (s2 == s).any()])
            boot[k] = centroid_dist(bf1, bf2) if len(bf1) and len(bf2) else np.nan
        rows.append({
            'ph1': ph1, 'ph2': ph2,
            'metric': metric,
            'obs_distance': obs,
            'ci_lo': float(np.nanpercentile(boot, 2.5)),
            'ci_hi': float(np.nanpercentile(boot, 97.5)),
        })
    return rows


def nearest_centroid_loso_indexed(feats: np.ndarray, labels: np.ndarray,
                                   speakers: np.ndarray,
                                   orig_indices: np.ndarray) -> list:
    """Like nearest_centroid_loso but returns (orig_row_idx, true, pred) triples."""
    unique_spk = np.unique(speakers)
    records = []
    for spk in unique_spk:
        test_mask  = speakers == spk
        train_mask = ~test_mask
        X_train = feats[train_mask]; y_train = labels[train_mask]
        X_test  = feats[test_mask];  y_test  = labels[test_mask]
        idx_test = orig_indices[test_mask]

        unique_cls = np.unique(y_train)
        centroids  = np.stack([X_train[y_train == c].mean(axis=0)
                                for c in unique_cls])
        D_test   = cdist(X_test, centroids, metric='cosine')
        pred_idx = D_test.argmin(axis=1)
        y_pred   = unique_cls[pred_idx]
        for idx, yt, yp in zip(idx_test, y_test, y_pred):
            records.append((int(idx), str(yt), str(yp)))
    return records


def build_pred_map(records: list) -> dict:
    """Map orig_row_index → (true_label, pred_label)."""
    return {idx: (true, pred) for idx, true, pred in records}


def run_classifiers(df_ac: pd.DataFrame, npz_w, npz_x,
                    vowels: list, n_perm: int, seed: int) -> tuple:
    rows = []

    # Acoustic (all + L1/L2 sub-groups)
    df_v = df_ac[df_ac['phoneme'].isin(vowels)].dropna(subset=['F1_lob', 'F2_lob'])
    feats_ac = df_v[['F1_lob', 'F2_lob']].values
    labels   = df_v['phoneme'].values
    spks     = df_v['speaker_id'].values
    l1_ac    = df_v['L1'].values
    res_ac   = nearest_centroid_loso(feats_ac, labels, spks)
    rows.append({'representation': 'acoustic (F1+F2)_all',
                 'accuracy': res_ac['accuracy'],
                 'macro_f1': np.mean(list(res_ac['f1_per_class'].values()))})
    for grp_label, grp_mask in [('L1', l1_ac == 'fr'), ('L2', l1_ac == 'ru')]:
        sub_feats  = feats_ac[grp_mask]
        sub_labels = labels[grp_mask]
        sub_spks   = spks[grp_mask]
        if len(np.unique(sub_labels)) < 2:
            continue
        res_sub = nearest_centroid_loso(sub_feats, sub_labels, sub_spks)
        rows.append({'representation': f'acoustic (F1+F2)_{grp_label}',
                     'accuracy': res_sub['accuracy'],
                     'macro_f1': np.mean(list(res_sub['f1_per_class'].values()))})

    conf_dict = {'acoustic': res_ac}

    for label, npz_path in [('whisper', npz_w), ('xlsr', npz_x)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data   = np.load(npz_path, allow_pickle=True)
        layers = data['layers'].tolist()
        L      = layers[0]
        ph_arr = data['phoneme'].astype(str)
        # Prefer high-dim PCA (50) for classification; fall back to PCA-2
        pca_keys = [k for k in data.keys() if k.startswith(f'pca_') and k.endswith(f'_layer_{L}') and 'explained' not in k]
        pca_keys = sorted(pca_keys, key=lambda k: int(k.split('_')[1]), reverse=True)
        feats  = data[pca_keys[0]] if pca_keys else data[f'pca_2_layer_{L}']
        spk_arr = data['speaker_id'].astype(str)
        l1_arr  = data['L1'].astype(str)
        mask   = np.isin(ph_arr, vowels)
        mask  &= ~np.isnan(feats).any(axis=1)

        for grp_label, grp_mask in [('all', np.ones(mask.sum(), dtype=bool)),
                                     ('L1', l1_arr[mask] == 'fr'),
                                     ('L2', l1_arr[mask] == 'ru')]:
            sub_feats = feats[mask][grp_mask]
            sub_labels = ph_arr[mask][grp_mask]
            sub_spks   = spk_arr[mask][grp_mask]
            if len(np.unique(sub_labels)) < 2:
                continue
            res = nearest_centroid_loso(sub_feats, sub_labels, sub_spks)
            rows.append({'representation': f'{label}_layer{L}_{grp_label}',
                         'accuracy': res['accuracy'],
                         'macro_f1': np.mean(list(res['f1_per_class'].values()))})
            if grp_label == 'all':
                conf_dict[label] = res

    return pd.DataFrame(rows), conf_dict


# ── Main ─────────────────────────────────────────────────────────────────────

def main(acoustic_norm, whisper_pca, xlsr_pca,
         out_group_comp, out_gender_residual,
         out_dist_mantel, out_pairwise_ci,
         out_classif, out_mcnemar, out_conf_fig,
         n_perm, bootstrap_n, french_vowels, seed):

    for p in [out_group_comp, out_gender_residual, out_dist_mantel,
              out_pairwise_ci, out_classif, out_mcnemar]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(os.path.dirname(out_conf_fig), exist_ok=True)

    df = pd.read_csv(acoustic_norm)

    # ── 6.1 Acoustic group comparisons ───────────────────────────────────────
    print("L1/L2 acoustic tests …")
    df_gc_ac = test_l1_l2_acoustic(df, french_vowels)

    # ── 6.1 Neural permutation tests ─────────────────────────────────────────
    print("L1/L2 neural permutation tests …")
    df_gc_w = test_l1_l2_neural(whisper_pca, french_vowels, 4, n_perm, seed)
    df_gc_x = test_l1_l2_neural(xlsr_pca,   french_vowels, 3, n_perm, seed)
    if not df_gc_w.empty:
        df_gc_w.insert(0, 'representation', 'whisper')
    if not df_gc_x.empty:
        df_gc_x.insert(0, 'representation', 'xlsr')

    gc_combined = pd.concat([df_gc_ac.assign(representation='acoustic'),
                              df_gc_w, df_gc_x], ignore_index=True)
    gc_combined.to_csv(out_group_comp, index=False)
    print(f"  Saved {out_group_comp}")

    # ── 6.1 Gender residual test (paired across phonemes) ───────────────────
    print("Gender residual paired test …")
    df_gender = gender_residual_test(df, french_vowels)
    df_gender.to_csv(out_gender_residual, index=False)
    print(f"  Saved {out_gender_residual}")

    # ── 6.2 Distance matrices + Mantel ───────────────────────────────────────
    print("Distance matrices + Mantel …")
    df_dm = distance_matrices_mantel(df, whisper_pca, xlsr_pca,
                                     french_vowels, n_perm, seed)
    df_dm.to_csv(out_dist_mantel, index=False)
    print(f"  Saved {out_dist_mantel}")

    # ── Pairwise phoneme distance bootstrap CIs (speaker-level) ─────────────
    print("Pairwise distance bootstrap CIs …")
    pw_rows = []
    for rep_label, npz_path in [('whisper', whisper_pca), ('xlsr', xlsr_pca)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path, allow_pickle=True)
        L       = data['layers'].tolist()[0]
        feats   = data[f'layer_{L}']
        ph_arr  = data['phoneme'].astype(str)
        spk_arr = data['speaker_id'].astype(str)
        valid   = ~np.isnan(feats).any(axis=1)
        feats, ph_arr, spk_arr = feats[valid], ph_arr[valid], spk_arr[valid]
        for row in pairwise_distance_bootstrap_ci(feats, ph_arr, spk_arr,
                                                   PHONEME_PAIRS, 'cosine',
                                                   bootstrap_n, seed):
            row['representation'] = f'{rep_label}_layer{L}'
            pw_rows.append(row)
    # Acoustic: Euclidean distance on Lobanov z-scores (speaker-level resampling)
    df_v      = df[df['phoneme'].isin(french_vowels)].dropna(subset=['F1_lob', 'F2_lob'])
    ac_feats  = df_v[['F1_lob', 'F2_lob']].values
    ac_ph_arr = df_v['phoneme'].values
    ac_sp_arr = df_v['speaker_id'].values
    for row in pairwise_distance_bootstrap_ci(ac_feats, ac_ph_arr, ac_sp_arr,
                                               PHONEME_PAIRS, 'euclidean',
                                               bootstrap_n, seed):
        row['representation'] = 'acoustic'
        pw_rows.append(row)
    df_pw = pd.DataFrame(pw_rows)
    df_pw.to_csv(out_pairwise_ci, index=False)
    print(f"  Saved {out_pairwise_ci}")

    # ── Nearest-centroid classifier ───────────────────────────────────────────
    print("Classifiers …")
    df_clf, conf_dict = run_classifiers(df, whisper_pca, xlsr_pca,
                                        french_vowels, n_perm, seed)
    df_clf.to_csv(out_classif, index=False)
    print(f"  Saved {out_classif}")

    # ── McNemar test (token-index-aligned matched pairs) ─────────────────────
    print("McNemar tests …")
    # Build pred_map: orig_row_idx → (true, pred) for each representation.
    # Row i in features_acoustic_norm.csv == row i in the NPZ arrays (same
    # phonemes.csv origin), so using the dataframe/array row index as a shared
    # token ID gives exact matched pairs across modalities.
    pred_maps = {}

    df_v_ac     = df[df['phoneme'].isin(french_vowels)].dropna(subset=['F1_lob','F2_lob'])
    ac_orig_idx = df_v_ac.index.values.astype(int)
    ac_recs     = nearest_centroid_loso_indexed(
        df_v_ac[['F1_lob', 'F2_lob']].values,
        df_v_ac['phoneme'].values,
        df_v_ac['speaker_id'].values,
        ac_orig_idx)
    pred_maps['acoustic'] = build_pred_map(ac_recs)

    for rep_label, npz_path in [('whisper', whisper_pca), ('xlsr', xlsr_pca)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path, allow_pickle=True)
        L       = data['layers'].tolist()[0]
        ph_all  = data['phoneme'].astype(str)
        spk_all = data['speaker_id'].astype(str)
        pca_keys = sorted(
            [k for k in data.keys() if k.startswith('pca_')
             and k.endswith(f'_layer_{L}') and 'explained' not in k],
            key=lambda k: int(k.split('_')[1]), reverse=True)
        feats_all = data[pca_keys[0]] if pca_keys else data[f'pca_2_layer_{L}']
        mask      = np.isin(ph_all, french_vowels) & ~np.isnan(feats_all).any(axis=1)
        orig_idx  = np.where(mask)[0]
        nn_recs   = nearest_centroid_loso_indexed(
            feats_all[mask], ph_all[mask], spk_all[mask], orig_idx)
        pred_maps[f'{rep_label}_layer{L}'] = build_pred_map(nn_recs)

    mcnemar_rows = []
    keys = list(pred_maps.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            m1, m2  = pred_maps[k1], pred_maps[k2]
            common  = sorted(set(m1.keys()) & set(m2.keys()))
            n_common = len(common)
            if n_common == 0:
                continue
            t1 = np.array([m1[i][0] for i in common])
            p1 = np.array([m1[i][1] for i in common])
            t2 = np.array([m2[i][0] for i in common])
            p2 = np.array([m2[i][1] for i in common])
            result = mcnemar_test(t1, p1, t2, p2)
            mcnemar_rows.append({'clf1': k1, 'clf2': k2,
                                  'n_matched_tokens': n_common, **result})
    pd.DataFrame(mcnemar_rows).to_csv(out_mcnemar, index=False)
    print(f"  Saved {out_mcnemar}")

    # ── Confusion matrix plots ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(conf_dict), figsize=(5 * len(conf_dict), 5))
    if len(conf_dict) == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, conf_dict.items()):
        plot_confusion_matrix(res['confusion'],
                              list(res['classes']),
                              f'{label} (acc={res["accuracy"]:.2f})', ax)
    fig.tight_layout()
    fig.savefig(out_conf_fig, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_conf_fig}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        acoustic_norm        = snakemake.input.acoustic_norm,
        whisper_pca          = snakemake.input.whisper_pca,
        xlsr_pca             = snakemake.input.xlsr_pca,
        out_group_comp       = snakemake.output.group_comparisons,
        out_gender_residual  = snakemake.output.gender_residual,
        out_dist_mantel      = snakemake.output.distance_mantel,
        out_pairwise_ci      = snakemake.output.pairwise_distance_ci,
        out_classif          = snakemake.output.classifier_results,
        out_mcnemar          = snakemake.output.mcnemar_results,
        out_conf_fig         = snakemake.output.confusion_matrices,
        n_perm               = snakemake.params.n_permutations,
        bootstrap_n          = snakemake.params.bootstrap_n,
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
        out_group_comp       = f"{r}/tables/group_comparisons.csv",
        out_gender_residual  = f"{r}/tables/gender_residual.csv",
        out_dist_mantel      = f"{r}/tables/distance_matrices_mantel.csv",
        out_pairwise_ci      = f"{r}/tables/pairwise_distance_ci.csv",
        out_classif          = f"{r}/tables/classifier_results.csv",
        out_mcnemar          = f"{r}/tables/mcnemar_results.csv",
        out_conf_fig         = f"{r}/figures/confusion_matrices.png",
        n_perm               = cfg['n_permutations'],
        bootstrap_n          = cfg['bootstrap_n'],
        french_vowels        = cfg['french_oral_vowels'],
        seed                 = cfg['random_seed'],
    )
