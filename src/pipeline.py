# Pipeline preprocessing audio bersama untuk seluruh model ChickVoice-EFEST
# Alur: WAV -> Resample 22.050 Hz -> 40 MFCC -> Crop/Pad 200 frame
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import librosa


# --- Ekstraksi Fitur MFCC Satuan ---
def extract_mfcc(
    path: str,
    sample_rate: int,
    n_mfcc: int,
    max_frames: int,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    random_seed: int = 42,
) -> np.ndarray:
    # Memuat berkas audio mono
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    # Augmentasi white-noise untuk variasi data training
    if augment:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            if rng is None:
                rng = np.random.default_rng(random_seed)
            scale = 0.005 * peak * float(rng.uniform(0.0, 1.0))
            audio = np.clip(
                audio + scale * rng.normal(size=audio.shape), -1.0, 1.0
            )

    # Ekstraksi koefisien MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    # Penyesuaian panjang frame (padding jika kurang, potong jika lebih)
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_frames - mfcc.shape[1])),
            mode="constant",
        )
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc.T.astype(np.float32)


# --- Pembuatan Batch Fitur Dataset ---
def make_arrays(
    frame: pd.DataFrame,
    classes: Sequence[str],
    sample_rate: int,
    n_mfcc: int,
    max_frames: int,
    training: bool,
    artifact_dir=None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    errors: list[dict] = []

    # Iterasi setiap sampel audio pada split
    for row in frame.itertuples(index=False):
        try:
            # Sampel asli tanpa augmentasi
            xs.append(
                extract_mfcc(
                    row.path, sample_rate, n_mfcc, max_frames,
                    augment=False,
                )
            )
            ys.append(list(classes).index(row.label))

            # Sampel tambahan dengan augmentasi khusus split training
            if training:
                xs.append(
                    extract_mfcc(
                        row.path, sample_rate, n_mfcc, max_frames,
                        augment=True, rng=rng,
                    )
                )
                ys.append(list(classes).index(row.label))
        except Exception as exc:
            errors.append(
                {"path": row.path, "label": row.label, "error": str(exc)}
            )
            print(f"[WARN] Skipping {row.path}: {exc}")

    # Catat error ekstraksi jika ada ke berkas CSV
    if errors and artifact_dir is not None:
        pd.DataFrame(errors).to_csv(
            artifact_dir / "feature_extraction_errors.csv", index=False
        )

    if not xs:
        raise RuntimeError("No audio features could be extracted.")

    return np.stack(xs), np.asarray(ys, dtype=np.int64)
