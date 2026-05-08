"""
Stage 1 – parse_corpus
Parse TextGrid files + metadata to produce one row per phoneme token.

Output columns:
  speaker_id, sentence_id, target_word, rep_index, phoneme,
  onset, offset, duration_ms, L1, gender, wav_path
"""

import os
import re
import sys

import pandas as pd
import yaml


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def parse_textgrid_phones(path: str) -> list[tuple]:
    """Return list of (xmin, xmax, text) for non-empty phone intervals."""
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip() for l in f]

    # Locate the "phones" tier
    phones_start = None
    for i, line in enumerate(lines):
        if re.match(r'^\s*name = "phones"\s*$', line):
            phones_start = i
            break
    if phones_start is None:
        return []

    intervals: list[dict] = []
    curr: dict = {}
    in_intervals = False

    for line in lines[phones_start + 1:]:
        line = line.strip()

        # Next item tier → stop
        if re.match(r'^item \[\d+\]:$', line):
            break

        if re.match(r'^intervals \[\d+\]:$', line):
            if curr:
                intervals.append(curr)
            curr = {}
            in_intervals = True
            continue

        if not in_intervals:
            continue

        m = re.match(r'^xmin = (.+)$', line)
        if m:
            curr['xmin'] = float(m.group(1))
            continue
        m = re.match(r'^xmax = (.+)$', line)
        if m:
            curr['xmax'] = float(m.group(1))
            continue
        m = re.match(r'^text = "(.*)"$', line)
        if m:
            curr['text'] = m.group(1)
            continue

    if curr:
        intervals.append(curr)

    SILENCE = {'', 'sp', 'sil', 'SIL', 'SP', '<sil>', '<sp>'}
    return [
        (d['xmin'], d['xmax'], d['text'])
        for d in intervals
        if d.get('text', '').strip() not in SILENCE
    ]


def canonical_phoneme(raw: str) -> str:
    """Strip corpus-specific prefixes and normalise phoneme labels."""
    # "ding... d" → "d"  (annotator disfluency marker)
    raw = re.sub(r'^ding\.+\s*', '', raw.strip())
    # Remove length/diacritic markers for analysis variants
    # (keep ː, ̃ as they are phonemically relevant in French)
    return raw.strip()


def load_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path, sep=';')
    meta.columns = [c.strip() for c in meta.columns]
    meta['spk']    = meta['spk'].str.strip()
    meta['L1']     = meta['L1'].str.strip()
    meta['Gender'] = meta['Gender'].str.strip()
    return meta.set_index('spk')


def load_rufrcorr(path: str) -> dict[int, tuple]:
    """Return {sentence_id: (target_word, rep_index 1-6)}."""
    df = pd.read_csv(path, sep='\t')
    sent_to_info: dict[int, tuple] = {}
    occ_cols = ['occ.1', 'occ.2', 'occ.3', 'occ.4', 'occ.5', 'occ.6']
    for _, row in df.iterrows():
        word = row['Word']
        for rep_idx, col in enumerate(occ_cols, 1):
            if col in row.index and not pd.isna(row[col]):
                sid = int(row[col])
                if sid not in sent_to_info:
                    sent_to_info[sid] = (word, rep_idx)
    return sent_to_info


# ── Main ─────────────────────────────────────────────────────────────────────

def main(corpus_dir, metadata_file, rufrcorr_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    meta         = load_metadata(metadata_file)
    sent_to_info = load_rufrcorr(rufrcorr_file)

    rows = []

    for spk in sorted(os.listdir(corpus_dir)):
        if spk.startswith('.'):
            continue
        spk_dir = os.path.join(corpus_dir, spk)
        if not os.path.isdir(spk_dir):
            continue

        if spk not in meta.index:
            print(f"Warning: {spk} not in metadata — skipping", file=sys.stderr)
            continue

        l1     = meta.loc[spk, 'L1']
        gender = meta.loc[spk, 'Gender']

        for fname in sorted(os.listdir(spk_dir)):
            if not fname.endswith('.TextGrid'):
                continue

            m = re.search(r'FRcorp(\d+)\.TextGrid$', fname)
            if not m:
                continue
            sent_id = int(m.group(1))

            tg_path  = os.path.join(spk_dir, fname)
            wav_path = tg_path.replace('.TextGrid', '.wav')

            if not os.path.exists(wav_path):
                print(f"Warning: WAV missing: {wav_path}", file=sys.stderr)
                continue

            target_word, rep_index = sent_to_info.get(sent_id, (None, None))
            phones = parse_textgrid_phones(tg_path)

            for xmin, xmax, phoneme in phones:
                rows.append({
                    'speaker_id':       spk,
                    'sentence_id':      sent_id,
                    'target_word':      target_word,
                    'rep_index':        rep_index,
                    'phoneme_raw':      phoneme,
                    'phoneme':          canonical_phoneme(phoneme),
                    'onset':       round(xmin, 6),
                    'offset':      round(xmax, 6),
                    'duration_ms': round((xmax - xmin) * 1000, 3),
                    'L1':          l1,
                    'gender':      gender,
                    'wav_path':    os.path.abspath(wav_path),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"Wrote {len(df):,} phoneme tokens → {output_file}")
    print(f"Speakers : {df['speaker_id'].nunique()}")
    l1_dist = df.groupby('L1')['speaker_id'].nunique().to_dict()
    g_dist  = df.groupby('gender')['speaker_id'].nunique().to_dict()
    print(f"L1 status: {l1_dist}")
    print(f"Gender   : {g_dist}")
    print(f"Phonemes : {df['phoneme'].nunique()} unique labels")
    print(f"Sample phoneme labels: {sorted(df['phoneme'].unique())[:20]}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        corpus_dir    = snakemake.params.corpus_dir,
        metadata_file = snakemake.input.metadata,
        rufrcorr_file = snakemake.input.rufrcorr,
        output_file   = snakemake.output.phonemes,
    )
except NameError:
    cfg = _load_config()
    main(
        corpus_dir    = cfg["corpus_dir"],
        metadata_file = cfg["metadata_file"],
        rufrcorr_file = cfg["rufrcorr_file"],
        output_file   = os.path.join(cfg["data_dir"], "phonemes.csv"),
    )
