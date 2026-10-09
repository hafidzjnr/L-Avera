"""Audit WAV files before splitting or training."""
from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import pandas as pd

from experiment.Baseline.config import ARTIFACT_DIR, CLASSES, DATASET_DIR, MANIFEST_PATH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR.resolve()}\n"
            "Update DATASET_DIR in Baseline/config.py."
        )

    rows = []
    for label in CLASSES:
        folder = DATASET_DIR / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        for path in paths:
            row = {
                "path": str(path.as_posix()),
                "label": label,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "status": "ok",
                "error": "",
            }

            try:
                audio, sample_rate = librosa.load(path, sr=None, mono=False)
                if audio.size == 0:
                    raise ValueError("empty audio")
                row["sample_rate"] = int(sample_rate)
                row["duration_seconds"] = float(audio.shape[-1] / sample_rate)
                row["channels"] = 1 if audio.ndim == 1 else int(audio.shape[0])
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No WAV files found under {DATASET_DIR}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"Manifest: {MANIFEST_PATH}")
    print("\nFile count by class:")
    print(
        manifest.groupby("label")
        .size()
        .reindex(CLASSES, fill_value=0)
        .to_string()
    )
    print(f"\nTotal files: {len(manifest)}")
    print(f"Unreadable/empty files: {(manifest['status'] != 'ok').sum()}")

    duplicates = manifest[
        manifest.duplicated("sha256", keep=False)
    ].sort_values("sha256")
    if not duplicates.empty:
        duplicate_report = ARTIFACT_DIR / "possible_exact_duplicates.csv"
        duplicates.to_csv(duplicate_report, index=False)
        print(f"WARNING: exact duplicate contents found: {duplicate_report}")

    sample_rates = sorted(
        manifest.loc[manifest["status"] == "ok", "sample_rate"]
        .dropna()
        .unique()
    )
    print(f"\nOriginal sample rates: {sample_rates}")
    print("\nReview the manifest and any exact-duplicate report before splitting.")


if __name__ == "__main__":
    main()
