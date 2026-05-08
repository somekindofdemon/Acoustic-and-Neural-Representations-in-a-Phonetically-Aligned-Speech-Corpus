"""
Stage 4 – extract_neural_xlsr
Average-pool hidden states from three XLS-R (wav2vec2-large-xlsr-53) layers
over the phoneme interval.

XLS-R feature-extractor strides: 5,2,2,2,2,2,2 → total 320 → 50 frames/s
(same as Whisper encoder after its conv layers).

Output: data/features_xlsr.npz  (same schema as features_whisper.npz)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

_XLSR_SR         = 16_000
_XLSR_FEAT_STRIDE = 320    # samples per frame
_XLSR_FRAME_RATE  = _XLSR_SR / _XLSR_FEAT_STRIDE   # 50 Hz


def _load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _time_to_frames(onset, offset, n_frames):
    start = int(onset  * _XLSR_FRAME_RATE)
    end   = max(int(offset * _XLSR_FRAME_RATE), start + 1)
    start = min(start, n_frames - 1)
    end   = min(end,   n_frames)
    if start >= end:
        end = start + 1
    return slice(start, end)


def extract_file(model, feature_extractor, wav_path: str,
                 group: pd.DataFrame, layers: list, device: str) -> dict:
    import librosa

    audio, _ = librosa.load(wav_path, sr=_XLSR_SR, mono=True)
    inp = feature_extractor(audio, sampling_rate=_XLSR_SR,
                            return_tensors="pt",
                            padding=False).input_values.to(device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True)

    # hidden_states: tuple[layer+1], each (1, T, D)
    hs_all   = out.hidden_states
    n_frames = hs_all[0].shape[1]

    results = {L: [] for L in layers}
    for _, row in group.iterrows():
        sl = _time_to_frames(row['onset'], row['offset'], n_frames)
        for L in layers:
            vec = hs_all[L][0, sl, :].mean(dim=0).cpu().float().numpy()
            results[L].append(vec)
    return results


def main(phonemes_file, output_file, model_name, layers):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {model_name} …")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model             = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    phonemes = pd.read_csv(phonemes_file)
    n_files  = phonemes['wav_path'].nunique()

    d_model   = model.config.hidden_size
    layer_feats  = {L: [] for L in layers}
    token_indices = []
    meta_cols = {c: [] for c in ['phoneme', 'speaker_id', 'sentence_id', 'L1', 'gender']}

    for i, (wav_path, grp) in enumerate(phonemes.groupby('wav_path', sort=False)):
        if i % 100 == 0:
            print(f"  XLS-R {i}/{n_files} …", flush=True)
        try:
            res = extract_file(model, feature_extractor, wav_path, grp, layers, device)
            for L in layers:
                layer_feats[L].extend(res[L])
            token_indices.extend(grp.index.tolist())
            for c in meta_cols:
                meta_cols[c].extend(grp[c].tolist())
        except Exception as e:
            print(f"Error {wav_path}: {e}", file=sys.stderr)
            dummy = np.full(d_model, np.nan, dtype=np.float32)
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
        layers        = list(snakemake.params.layers),
    )
except NameError:
    cfg = _load_config()
    main(
        phonemes_file = os.path.join(cfg['data_dir'], 'phonemes.csv'),
        output_file   = os.path.join(cfg['data_dir'], 'features_xlsr.npz'),
        model_name    = cfg['xlsr_model'],
        layers        = cfg['xlsr_layers'],
    )
