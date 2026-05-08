"""
Section 7 – Linear Mixed-Effects Models

Model sequence (acoustic F1/F2, and neural PC1..PC5):
  0. Null    : F ~ 1 + (1|speaker)
  1. Main    : F ~ L2 + Male + (1|speaker)
  2. Full    : F ~ L2 * Male + (1|speaker)
  3. Extended: F ~ L2 * Male + vowel_height + (1|speaker)
  4. Slope   : F ~ L2 * Male + (1|speaker) + (L2|speaker)

Metrics reported per model: AIC, BIC, logLik, LRT vs previous, ICC (null),
marginal R² and conditional R² (Nakagawa & Schielzeth 2013).

Output: results/tables/lme_results.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMParams
import yaml

warnings.filterwarnings("ignore")

VOWEL_HEIGHT = {
    'i': 'high', 'y': 'high', 'u': 'high',
    'e': 'mid',  'ø': 'mid',  'o': 'mid',
    'ɛ': 'mid',  'œ': 'mid',  'ɔ': 'mid',
    'a': 'low',  'ɑ': 'low',  'ə': 'mid',
}


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── R² (Nakagawa & Schielzeth 2013) ─────────────────────────────────────────

def r2_nakagawa(model_result) -> tuple:
    """Return (marginal_R2, conditional_R2) for a MixedLM result."""
    fe = model_result.fe_params
    X  = model_result.model.exog
    var_fixed = float(np.var(X @ fe.values, ddof=0))

    var_random = float(model_result.cov_re.values[0, 0]) \
        if model_result.cov_re is not None else 0.0

    var_resid  = float(model_result.scale)

    denom = var_fixed + var_random + var_resid
    if denom == 0:
        return np.nan, np.nan
    R2_m = var_fixed / denom
    R2_c = (var_fixed + var_random) / denom
    return R2_m, R2_c


# ── ICC from null model ───────────────────────────────────────────────────────

def icc_from_null(null_result) -> float:
    var_u = float(null_result.cov_re.values[0, 0])
    var_e = float(null_result.scale)
    return var_u / (var_u + var_e) if (var_u + var_e) > 0 else np.nan


# ── LRT between nested models ─────────────────────────────────────────────────

def lrt(model_full, model_nested) -> tuple:
    """Likelihood-ratio test. Both models must be fitted with ML (not REML)."""
    from scipy.stats import chi2
    ll_full   = model_full.llf
    ll_nested = model_nested.llf
    df_diff   = (len(model_full.params) - len(model_nested.params))
    if df_diff <= 0:
        return np.nan, np.nan
    stat  = 2 * (ll_full - ll_nested)
    p_val = chi2.sf(stat, df_diff)
    return float(stat), float(p_val)


# ── Fit model sequence ────────────────────────────────────────────────────────

def fit_model_sequence(df: pd.DataFrame, response: str,
                       phoneme: str) -> list[dict]:
    sub = df[df['phoneme'] == phoneme].dropna(subset=[response]).copy()
    if len(sub) < 30 or sub['speaker_id'].nunique() < 4:
        return []

    sub['L2']   = (sub['L1'] == 'ru').astype(int)
    sub['Male'] = (sub['gender'] == 'm').astype(int)
    sub['vowel_height'] = sub['phoneme'].map(VOWEL_HEIGHT).fillna('mid')

    grp_kw  = dict(groups=sub['speaker_id'])
    fit_ml  = dict(reml=False, disp=False)   # ML for LRT comparisons
    fit_rem = dict(reml=True,  disp=False)   # REML for parameter estimates

    rows = []
    try:
        # 0. Null
        m0 = smf.mixedlm(f"{response} ~ 1", sub, **grp_kw).fit(**fit_ml)
        icc  = icc_from_null(m0)
        r2m, r2c = r2_nakagawa(m0)
        rows.append(_row('intercept_only', response, phoneme, m0, None, None, icc, r2m, r2c))

        # 1. Main effects
        m1 = smf.mixedlm(f"{response} ~ L2 + Male", sub, **grp_kw).fit(**fit_ml)
        lrt_stat, lrt_p = lrt(m1, m0)
        r2m, r2c = r2_nakagawa(m1)
        rows.append(_row('main', response, phoneme, m1, lrt_stat, lrt_p, icc, r2m, r2c))

        # 2. Full (interaction)
        m2 = smf.mixedlm(f"{response} ~ L2 * Male", sub, **grp_kw).fit(**fit_ml)
        lrt_stat, lrt_p = lrt(m2, m1)
        r2m, r2c = r2_nakagawa(m2)
        rows.append(_row('full', response, phoneme, m2, lrt_stat, lrt_p, icc, r2m, r2c))

        # 3. Extended (vowel height)
        try:
            m3 = smf.mixedlm(f"{response} ~ L2 * Male + C(vowel_height)", sub,
                              **grp_kw).fit(**fit_ml)
            lrt_stat, lrt_p = lrt(m3, m2)
            r2m, r2c = r2_nakagawa(m3)
            rows.append(_row('extended', response, phoneme, m3, lrt_stat, lrt_p, icc, r2m, r2c))
        except Exception:
            pass

        # 4. Random slope (L2|speaker)
        try:
            m4 = smf.mixedlm(f"{response} ~ L2 * Male", sub,
                              groups=sub['speaker_id'],
                              re_formula='~L2').fit(**fit_ml)
            lrt_stat, lrt_p = lrt(m4, m2)
            r2m, r2c = r2_nakagawa(m4)
            rows.append(_row('random_slope', response, phoneme, m4, lrt_stat, lrt_p, icc, r2m, r2c))
        except Exception:
            pass

    except Exception as e:
        print(f"    LME failed ({response}, {phoneme}): {e}", file=sys.stderr)

    return rows


def _row(model_name, response, phoneme, result, lrt_stat, lrt_p,
         icc, R2m, R2c) -> dict:
    fe = result.fe_params.to_dict()
    return {
        'model':    model_name,
        'response': response,
        'phoneme':  phoneme,
        'AIC':      result.aic,
        'BIC':      result.bic,
        'logLik':   result.llf,
        'LRT_stat': lrt_stat,
        'LRT_p':    lrt_p,
        'ICC':      icc,
        'R2_marginal':    R2m,
        'R2_conditional': R2c,
        **{f'fe_{k}': v for k, v in fe.items()},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(acoustic_norm, whisper_pca, xlsr_pca,
         out_lme, french_vowels):
    os.makedirs(os.path.dirname(out_lme), exist_ok=True)

    df_ac = pd.read_csv(acoustic_norm)
    all_rows = []

    # ── Acoustic models ───────────────────────────────────────────────────────
    print("Acoustic LME models …")
    for ph in french_vowels:
        for feat in ['F1_lob', 'F2_lob']:
            rows = fit_model_sequence(df_ac, feat, ph)
            all_rows.extend(rows)

    # ── Neural (PC-projected) models ──────────────────────────────────────────
    for rep_label, npz_path in [('whisper', whisper_pca), ('xlsr', xlsr_pca)]:
        if npz_path is None or not os.path.exists(npz_path):
            continue
        print(f"{rep_label} LME models …")
        data    = np.load(npz_path, allow_pickle=True)
        layers  = data['layers'].tolist()
        L       = layers[0]
        # Prefer high-dim PCA (pca_50) so we can take the first 5 PCs
        # as required by Section 7.2.  Fall back to pca_2 if unavailable.
        pca_keys = sorted(
            [k for k in data.keys() if k.startswith('pca_')
             and k.endswith(f'_layer_{L}') and 'explained' not in k],
            key=lambda k: int(k.split('_')[1]), reverse=True)
        if not pca_keys:
            continue
        pca_key = pca_keys[0]
        feats   = data[pca_key]
        ph_arr  = data['phoneme'].astype(str)
        spk_arr = data['speaker_id'].astype(str)
        l1_arr  = data['L1'].astype(str)
        gen_arr = data['gender'].astype(str)

        for pc_dim in range(min(5, feats.shape[1])):
            df_nn = pd.DataFrame({
                'phoneme':    ph_arr,
                'speaker_id': spk_arr,
                'L1':         l1_arr,
                'gender':     gen_arr,
                f'PC{pc_dim+1}': feats[:, pc_dim],
            })
            for ph in french_vowels:
                rows = fit_model_sequence(
                    df_nn, f'PC{pc_dim+1}', ph)
                for r in rows:
                    r['representation'] = f'{rep_label}_layer{L}'
                all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_lme, index=False)
    print(f"Saved {len(df_out):,} model rows → {out_lme}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        acoustic_norm = snakemake.input.acoustic_norm,
        whisper_pca   = snakemake.input.whisper_pca,
        xlsr_pca      = snakemake.input.xlsr_pca,
        out_lme       = snakemake.output.lme_results,
        french_vowels = list(snakemake.params.french_vowels),
    )
except NameError:
    cfg = _load_config()
    d, r = cfg['data_dir'], cfg['results_dir']
    main(
        acoustic_norm = f"{d}/features_acoustic_norm.csv",
        whisper_pca   = f"{d}/features_whisper_pca.npz",
        xlsr_pca      = f"{d}/features_xlsr_pca.npz",
        out_lme       = f"{r}/tables/lme_results.csv",
        french_vowels = cfg['french_oral_vowels'],
    )
