"""
Stage 2 – extract_acoustics
For each phoneme token extract F1, F2, F3 (midpoint), f0 (mean, voiced only),
and SCG (for fricatives).  For long vowels (> long_vowel_ms) also extract at
25% and 75%.

Output: data/features_acoustic.csv
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import yaml

VOWELS = set("aɑeɛiɔouøœəɑ̃ɛ̃ɔ̃")

def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ── Acoustic extraction helpers ───────────────────────────────────────────────

def get_formants_at(formant_obj, time: float, n: int = 3) -> list[Optional[float]]:
    vals = []
    for k in range(1, n + 1):
        v = formant_obj.get_value_at_time(k, time)
        vals.append(None if (v is None or np.isnan(v)) else v)
    return vals


def extract_token(sound: parselmouth.Sound, onset: float, offset: float,
                  gender: str, phoneme: str, params: dict) -> dict:
    """Extract acoustic features for one phoneme token."""
    max_f = params['max_formant_female'] if gender == 'f' else params['max_formant_male']
    n_f   = params['n_formants']

    seg = sound.extract_part(from_time=onset, to_time=offset, preserve_times=True)
    mid = (onset + offset) / 2.0

    # ── Formants at midpoint ──────────────────────────────────────────────────
    formant = seg.to_formant_burg(
        time_step=None,
        max_number_of_formants=n_f,
        maximum_formant=max_f,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )
    F1_mid, F2_mid, F3_mid = get_formants_at(formant, mid, n=3)

    # ── Long-vowel trajectory (25 % / 75 %) ──────────────────────────────────
    dur_ms = (offset - onset) * 1000.0
    F1_25 = F2_25 = F1_75 = F2_75 = None
    if dur_ms > params['long_vowel_ms'] and phoneme.strip() in VOWELS:
        t25 = onset + 0.25 * (offset - onset)
        t75 = onset + 0.75 * (offset - onset)
        F1_25, F2_25, _ = get_formants_at(formant, t25)
        F1_75, F2_75, _ = get_formants_at(formant, t75)

    # ── f0 (voiced segments) ──────────────────────────────────────────────────
    try:
        pitch = seg.to_pitch()
        f0_vals = [
            pitch.get_value_at_time(t)
            for t in np.linspace(onset, offset, 10)
        ]
        f0_vals = [v for v in f0_vals if v is not None and not np.isnan(v)]
        f0_mean = float(np.mean(f0_vals)) if f0_vals else None
    except Exception:
        f0_mean = None

    # ── Spectral centre of gravity ────────────────────────────────────────────
    try:
        spectrum = seg.to_spectrum()
        scg = call(spectrum, "Get centre of gravity", 2)
        scg = float(scg) if (scg is not None and not np.isnan(scg)) else None
    except Exception:
        scg = None

    return {
        'F1':    F1_mid,
        'F2':    F2_mid,
        'F3':    F3_mid,
        'F1_25': F1_25,
        'F2_25': F2_25,
        'F1_75': F1_75,
        'F2_75': F2_75,
        'f0':    f0_mean,
        'SCG':   scg,
    }


# ── Missing values report ─────────────────────────────────────────────────────

def missing_values_report(df: pd.DataFrame, out_path: str):
    """Per-phoneme × per-group (L1 × gender) missing F1/F2 counts."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for ph, ph_grp in df.groupby('phoneme'):
        for l1, l1_grp in ph_grp.groupby('L1'):
            for gender, g_grp in l1_grp.groupby('gender'):
                n = len(g_grp)
                for feat in ['F1', 'F2', 'F3', 'f0']:
                    if feat not in g_grp.columns:
                        continue
                    n_miss = g_grp[feat].isna().sum()
                    rows.append({
                        'phoneme': ph, 'L1': l1, 'gender': gender,
                        'feature': feat, 'n_total': n,
                        'n_missing': int(n_miss),
                        'pct_missing': round(100.0 * n_miss / n, 2) if n > 0 else 0.0,
                    })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Missing values report → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(phonemes_file, output_file, missing_values_file, params):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    phonemes = pd.read_csv(phonemes_file)
    results  = []

    # Process file by file to avoid redundant disk I/O
    grouped   = phonemes.groupby('wav_path', sort=False)
    n_files   = len(grouped)
    n_errors  = 0

    for i, (wav_path, grp) in enumerate(grouped):
        if i % 100 == 0:
            print(f"  Acoustics {i}/{n_files} …", flush=True)

        try:
            sound = parselmouth.Sound(wav_path)
        except Exception as e:
            print(f"Cannot load {wav_path}: {e}", file=sys.stderr)
            for _, row in grp.iterrows():
                results.append({**row.to_dict(),
                                 'F1': None, 'F2': None, 'F3': None,
                                 'F1_25': None, 'F2_25': None,
                                 'F1_75': None, 'F2_75': None,
                                 'f0': None, 'SCG': None})
            n_errors += len(grp)
            continue

        for _, row in grp.iterrows():
            try:
                feats = extract_token(
                    sound, row['onset'], row['offset'],
                    row['gender'], row['phoneme'], params
                )
            except Exception as e:
                feats = {'F1': None, 'F2': None, 'F3': None,
                         'F1_25': None, 'F2_25': None,
                         'F1_75': None, 'F2_75': None,
                         'f0': None, 'SCG': None}
                n_errors += 1
            results.append({**row.to_dict(), **feats})

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    n = len(df)
    miss_F1 = df['F1'].isna().sum()
    miss_F2 = df['F2'].isna().sum()
    print(f"Wrote {n:,} rows → {output_file}")
    print(f"Missing F1: {miss_F1:,} ({100*miss_F1/n:.1f}%)")
    print(f"Missing F2: {miss_F2:,} ({100*miss_F2/n:.1f}%)")
    if n_errors:
        print(f"Extraction errors: {n_errors:,}", file=sys.stderr)

    missing_values_report(df, missing_values_file)


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    p = {
        'max_formant_female': snakemake.params.max_formant_female,
        'max_formant_male':   snakemake.params.max_formant_male,
        'n_formants':         snakemake.params.n_formants,
        'long_vowel_ms':      snakemake.params.long_vowel_ms,
    }
    main(snakemake.input.phonemes, snakemake.output.features,
         snakemake.output.missing_values, p)
except NameError:
    cfg = _load_config()
    p = {
        'max_formant_female': cfg['max_formant_female'],
        'max_formant_male':   cfg['max_formant_male'],
        'n_formants':         cfg['n_formants'],
        'long_vowel_ms':      cfg['long_vowel_threshold_ms'],
    }
    main(
        os.path.join(cfg['data_dir'], 'phonemes.csv'),
        os.path.join(cfg['data_dir'], 'features_acoustic.csv'),
        os.path.join(cfg['results_dir'], 'tables', 'missing_values.csv'),
        p,
    )
