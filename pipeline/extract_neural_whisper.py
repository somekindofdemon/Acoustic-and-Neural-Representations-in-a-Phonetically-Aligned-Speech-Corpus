"""
Stage 3 – extract_neural_whisper
Average-pool hidden states from two Whisper encoder layers over the phoneme
interval.  Results are stored as a compressed npz with metadata arrays.

Output: data/features_whisper.npz
  token_index   : (N,) int   – row index into phonemes.csv
  layer_{L}     : (N, d)     – float32 vectors for encoder layer L
  phoneme       : (N,) str
  speaker_id    : (N,) str
  sentence_id   : (N,) int
  L1            : (N,) str
  gender        : (N,) str
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

# Whisper encoder: 2 × conv, then stride-2 overall → 50 frames/s at 16 kHz
_WHISPER_SR        = 16_000
_WHISPER_ENC_HOP   = 320   # samples per encoder frame (= mel_hop × 2 = 160 × 2)
_WHISPER_FRAME_RATE = _WHISPER_SR / _WHISPER_ENC_HOP  # 50 Hz


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _time_to_frames(onset, offset, n_frames):
    start = int(onset  * _WHISPER_FRAME_RATE)
    end   = max(int(offset * _WHISPER_FRAME_RATE), start + 1)
    start = min(start, n_frames - 1)
    end   = min(end,   n_frames)
    if start >= end:
        end = start + 1
    return slice(start, end)


def extract_file(model, processor, wav_path: str, group: pd.DataFrame,
                 layers: list, device: str) -> dict:
    """Return {layer: list of vectors} for all tokens in one WAV file."""
    import librosa

    audio, _ = librosa.load(wav_path, sr=_WHISPER_SR, mono=True)
    inp = processor(audio, sampling_rate=_WHISPER_SR,
                    return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        out = model.encoder(inp, output_hidden_states=True)

    # hidden_states: tuple[layer+1] each (1, T, D)
    hs_all = out.hidden_states  # index 0 = embedding, 1..N = transformer layers
    n_frames = hs_all[0].shape[1]

    results = {L: [] for L in layers}
    for _, row in group.iterrows():
        sl = _time_to_frames(row['onset'], row['offset'], n_frames)
        for L in layers:
            vec = hs_all[L][0, sl, :].mean(dim=0).cpu().float().numpy()
            results[L].append(vec)
    return results


def main(phonemes_file, output_file, model_name, layers):
    from transformers import WhisperProcessor, WhisperModel

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {model_name} …")

    processor = WhisperProcessor.from_pretrained(model_name)
    model     = WhisperModel.from_pretrained(model_name).to(device)
    model.eval()

    phonemes  = pd.read_csv(phonemes_file)
    n_files   = phonemes['wav_path'].nunique()

    layer_feats  = {L: [] for L in layers}
    token_indices = []
    meta_cols = {c: [] for c in ['phoneme', 'speaker_id', 'sentence_id', 'L1', 'gender']}

    for i, (wav_path, grp) in enumerate(phonemes.groupby('wav_path', sort=False)):
        if i % 100 == 0:
            print(f"  Whisper {i}/{n_files} …", flush=True)
        try:
            res = extract_file(model, processor, wav_path, grp, layers, device)
            for L in layers:
                layer_feats[L].extend(res[L])
            token_indices.extend(grp.index.tolist())
            for c in meta_cols:
                meta_cols[c].extend(grp[c].tolist())
        except Exception as e:
            print(f"Error {wav_path}: {e}", file=sys.stderr)
            dummy = np.full(model.config.d_model, np.nan, dtype=np.float32)
            for L in layers:
                layer_feats[L].extend([dummy] * len(grp))
            token_indices.extend(grp.index.tolist())
            for c in meta_cols:
                meta_cols[c].extend(grp[c].tolist())

    save = {
        'token_index': np.array(token_indices),
        'layers':      np.array(layers),
    }
    for L in layers:
        save[f'layer_{L}'] = np.stack(layer_feats[L])
    for c, vals in meta_cols.items():
        save[c] = np.array(vals, dtype=object)

    np.savez_compressed(output_file, **save)
    N = len(token_indices)
    print(f"Saved {N:,} vectors per layer {layers} → {output_file}")


# ── Snakemake / standalone entry ──────────────────────────────────────────────

try:
    main(
        phonemes_file = snakemake.input.phonemes,
        output_file   = snakemake.output.features,
        model_name    = snakemake.params.model,
        layers        = [snakemake.params.layer_lower, snakemake.params.layer_upper],
    )
except NameError:
    cfg = _load_config()
    main(
        phonemes_file = os.path.join(cfg['data_dir'], 'phonemes.csv'),
        output_file   = os.path.join(cfg['data_dir'], 'features_whisper.npz'),
        model_name    = cfg['whisper_model'],
        layers        = [cfg['whisper_layer_lower'], cfg['whisper_layer_upper']],
    )
