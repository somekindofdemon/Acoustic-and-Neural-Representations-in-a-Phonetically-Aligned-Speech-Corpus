"""
Section 8 – Confidence Intervals and ROPE

For acoustic: 95% profile-likelihood CIs on L1/L2 contrast in F1/F2
  (from selected LME model) → forest plot.
For neural  : 95% bootstrap CIs on cosine distance between L1/L2 centroids
  (speaker-level resampling) → forest plot.
ROPE classification: equivalent / non-equivalent / indeterminate.

Output
------
results/figures/forest_plots.png
results/tables/rope_classification.csv
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial.distance import cdist
import yaml

warnings.filterwarnings("ignore")


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── Bootstrap CI on cosine distance (speaker-level resampling) ───────────────

def bootstrap_cosine_ci(df_ph: pd.DataFrame, feats: np.ndarray,
                         spk_arr: np.ndarray, l1_arr: np.ndarray,
                         n_boot: int, seed: int) -> tuple:
    """
    Bootstrap 95% CI on cosine distance between L1 and L2 centroids.
    Resamples at the speaker level.
    """
    rng = np.random.default_rng(seed)
    speakers = np.unique(spk_arr)
    l1_spks  = np.array([spk for spk in speakers
                          if (l1_arr[spk_arr == spk][0] == 'fr')])
    l2_spks  = np.array([spk for spk in speakers
                          if (l1_arr[spk_arr == spk][0] == 'ru')])

    if len(l1_spks) < 2 or len(l2_spks) < 2:
        return np.nan, np.nan, np.nan

    def centroid_cosine(l1_s, l2_s):
        m1 = np.vstack([feats[spk_arr == s] for s in l1_s
                         if (spk_arr == s).sum() > 0])
        m2 = np.vstack([feats[spk_arr == s] for s in l2_s
                         if (spk_arr == s).sum() > 0])
        if len(m1) == 0 or len(m2) == 0:
            return np.nan
        c1 = m1.mean(axis=0)
        c2 = m2.mean(axis=0)
        return float(cdist(c1[None], c2[None], metric='cosine')[0, 0])

    obs  = centroid_cosine(l1_spks, l2_spks)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        bs_l1 = rng.choice(l1_spks, len(l1_spks), replace=True)
        bs_l2 = rng.choice(l2_spks, len(l2_spks), replace=True)
        boot[k] = centroid_cosine(bs_l1, bs_l2)

    ci_lo = float(np.nanpercentile(boot, 2.5))
    ci_hi = float(np.nanpercentile(boot, 97.5))
    return obs, ci_lo, ci_hi


# ── CIs from LME fixed effects ────────────────────────────────────────────────

def acoustic_cis_from_lme(df_ac: pd.DataFrame, vowels: list) -> pd.DataFrame:
    """Refit the 'full' model per vowel, extract CI on L2 coefficient."""
    rows = []
    for ph in vowels:
        for feat in ['F1_lob', 'F2_lob']:
            sub = df_ac[df_ac['phoneme'] == ph].dropna(subset=[feat]).copy()
            if len(sub) < 20 or sub['speaker_id'].nunique() < 4:
                continue
            sub['L2']   = (sub['L1'] == 'ru').astype(int)
            sub['Male'] = (sub['gender'] == 'm').astype(int)
            try:
                m = smf.mixedlm(f"{feat} ~ L2 * Male", sub,
                                  groups=sub['speaker_id']).fit(reml=True, disp=False)
                ci = m.conf_int()
                coef = float(m.fe_params['L2'])
                lo   = float(ci.loc['L2', 0])
                hi   = float(ci.loc['L2', 1])
                rows.append({
                    'phoneme': ph, 'feature': feat,
                    'estimate': coef, 'ci_lo': lo, 'ci_hi': hi,
                    'representation': 'acoustic',
                })
            except Exception:
                pass
    return pd.DataFrame(rows)


# ── Intra-speaker noise floor for neural ROPE ────────────────────────────────

def neural_noise_floor(npz_path: str, layer: int) -> float:
    """Mean intra-speaker cosine distance for the given layer."""
    data    = np.load(npz_path, allow_pickle=True)
    layers  = data['layers'].tolist()
    L       = layer if layer in layers else layers[0]
    feats   = data[f'layer_{L}']
    ph_arr  = data['phoneme'].astype(str)
    spk_arr = data['speaker_id'].astype(str)

    dists = []
    for spk in np.unique(spk_arr):
        mask = spk_arr == spk
        F    = feats[mask]
        F    = F[~np.isnan(F).any(axis=1)]
        if len(F) < 2:
            continue
        D = cdist(F, F, metric='cosine')
        idx = np.triu_indices(len(F), k=1)
        dists.extend(D[idx].tolist())
    return float(np.mean(dists)) if dists else 0.05


# ── ROPE classification ───────────────────────────────────────────────────────

def classify_rope(ci_lo, ci_hi, rope_lo, rope_hi) -> str:
    if ci_hi < rope_lo or ci_lo > rope_hi:
        return 'non-equivalent'
    if ci_lo >= rope_lo and ci_hi <= rope_hi:
        return 'equivalent'
    return 'indeterminate'


# ── Forest plot ───────────────────────────────────────────────────────────────

def forest_plot(df: pd.DataFrame, title: str, ax,
                rope_lo=None, rope_hi=None, zero_line=True):
    df = df.dropna(subset=['estimate', 'ci_lo', 'ci_hi']).reset_index(drop=True)
    y  = np.arange(len(df))

    ax.errorbar(df['estimate'], y,
                xerr=[np.abs(df['estimate'] - df['ci_lo']),
                      np.abs(df['ci_hi'] - df['estimate'])],
                fmt='o', color='steelblue', ecolor='steelblue',
                capsize=3, markersize=5, linewidth=1)

    if zero_line:
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    if rope_lo is not None:
        ax.axvspan(rope_lo, rope_hi, color='lightgreen', alpha=0.25,
                   label=f'ROPE [{rope_lo:.2f}, {rope_hi:.2f}]')
        ax.legend(fontsize=8)

    labels = (df['phoneme'] + ' ' + df.get('feature', pd.Series([''] * len(df)))).str.strip()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Estimate (95% CI)')
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(acoustic_norm, whisper_pca, xlsr_pca, lme_results,
         out_forest, out_rope,
         bootstrap_n, rope_hz, french_vowels, seed, rope_z=None):

    for p in [out_forest, out_rope]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    df_ac  = pd.read_csv(acoustic_norm)
    # ROPE in Lobanov z-score units (±20 Hz ≈ ±0.15 z given typical SD_F1 ≈ 130 Hz)
    _rope_z = rope_z if rope_z is not None else rope_hz / 130.0
    rope_acoustic = (-_rope_z, _rope_z)

    # ── 8.1 Acoustic CIs ─────────────────────────────────────────────────────
    print("Acoustic CIs …")
    df_ac_ci = acoustic_cis_from_lme(df_ac, french_vowels)

    # ── 8.2 Neural bootstrap CIs ──────────────────────────────────────────────
    print("Neural bootstrap CIs …")
    neural_rows = []
    rope_neural_dict = {}

    for rep_label, npz_path, default_layer in [('whisper', whisper_pca, 4),
                                                ('xlsr',    xlsr_pca,   3)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path, allow_pickle=True)
        layers  = data['layers'].tolist()
        L       = default_layer if default_layer in layers else layers[0]
        feats   = data[f'layer_{L}']
        ph_arr  = data['phoneme'].astype(str)
        spk_arr = data['speaker_id'].astype(str)
        l1_arr  = data['L1'].astype(str)

        # Noise floor for ROPE
        delta0 = neural_noise_floor(npz_path, L)
        rope_neural_dict[rep_label] = (0.0, delta0)
        print(f"  {rep_label} noise floor δ₀ = {delta0:.4f}")

        mask = ~np.isnan(feats).any(axis=1)
        feats_clean = feats[mask]
        ph_clean    = ph_arr[mask]
        spk_clean   = spk_arr[mask]
        l1_clean    = l1_arr[mask]

        for ph in french_vowels:
            ph_mask = ph_clean == ph
            if ph_mask.sum() < 4:
                continue
            obs, lo, hi = bootstrap_cosine_ci(
                None, feats_clean[ph_mask],
                spk_clean[ph_mask], l1_clean[ph_mask],
                bootstrap_n, seed)
            neural_rows.append({
                'phoneme': ph, 'representation': rep_label,
                'estimate': obs, 'ci_lo': lo, 'ci_hi': hi,
            })

    df_nn_ci = pd.DataFrame(neural_rows)

    # ── 8.4 ROPE classification ───────────────────────────────────────────────
    rope_rows = []

    for _, row in df_ac_ci.iterrows():
        cls = classify_rope(row['ci_lo'], row['ci_hi'], *rope_acoustic)
        rope_rows.append({**row.to_dict(),
                          'rope_lo': rope_acoustic[0],
                          'rope_hi': rope_acoustic[1],
                          'rope_class': cls})

    for _, row in df_nn_ci.iterrows():
        rep  = row['representation']
        rope = rope_neural_dict.get(rep, (0.0, 0.05))
        cls  = classify_rope(row['ci_lo'], row['ci_hi'], *rope)
        rope_rows.append({**row.to_dict(),
                          'rope_lo': rope[0],
                          'rope_hi': rope[1],
                          'rope_class': cls})

    df_rope = pd.DataFrame(rope_rows)
    df_rope.to_csv(out_rope, index=False)
    print(f"  Saved {out_rope}")

    # ── Forest plots ──────────────────────────────────────────────────────────
    n_panels = 1 + len(rope_neural_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(6, len(french_vowels) * 0.55 + 2)))

    # Acoustic forest
    df_f1 = df_ac_ci[df_ac_ci['feature'] == 'F1_lob'].copy()
    forest_plot(df_f1, 'Acoustic F1: L1/L2 contrast', axes[0],
                rope_lo=rope_acoustic[0], rope_hi=rope_acoustic[1])

    # Neural forests
    ax_idx = 1
    for rep_label in ['whisper', 'xlsr']:
        if ax_idx >= n_panels:
            break
        sub = df_nn_ci[df_nn_ci['representation'] == rep_label].copy()
        rope = rope_neural_dict.get(rep_label, (0.0, 0.05))
        if not sub.empty:
            forest_plot(sub, f'{rep_label}: L1/L2 cosine distance',
                        axes[ax_idx],
                        rope_lo=rope[0], rope_hi=rope[1], zero_line=False)
        ax_idx += 1

    fig.suptitle('L1/L2 contrasts with 95% CIs and ROPE', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_forest, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_forest}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        acoustic_norm = snakemake.input.acoustic_norm,
        whisper_pca   = snakemake.input.whisper_pca,
        xlsr_pca      = snakemake.input.xlsr_pca,
        lme_results   = snakemake.input.lme_results,
        out_forest    = snakemake.output.forest_plots,
        out_rope      = snakemake.output.rope_classification,
        bootstrap_n   = snakemake.params.bootstrap_n,
        rope_hz       = snakemake.params.rope_hz,
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
        lme_results   = f"{r}/tables/lme_results.csv",
        out_forest    = f"{r}/figures/forest_plots.png",
        out_rope      = f"{r}/tables/rope_classification.csv",
        bootstrap_n   = cfg['bootstrap_n'],
        rope_hz       = cfg['rope_acoustic_hz'],
        rope_z        = cfg.get('rope_acoustic_z', None),
        french_vowels = cfg['french_oral_vowels'],
        seed          = cfg['random_seed'],
    )
