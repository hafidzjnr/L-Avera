"""Create a reproducible train/validation/test split from original files."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from experiment.Baseline.config import (
    ARTIFACT_DIR, CLASSES, MANIFEST_PATH, RANDOM_SEED,
    SPLIT_PATH, TEST_FRACTION, VALIDATION_FRACTION,
)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Run first: python -m Baseline.audit_dataset")

    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["status"] == "ok"].copy()
    df = df[df["label"].isin(CLASSES)].copy()

    if df["path"].duplicated().any():
        raise ValueError("Duplicate paths in manifest; resolve before splitting.")
    if df["label"].nunique() != len(CLASSES):
        raise ValueError("One or more expected classes are missing.")

    # Exact byte-identical duplicates must be reviewed before split creation.
    duplicated_hashes = df[df.duplicated("sha256", keep=False)]
    if not duplicated_hashes.empty:
        raise ValueError(
            "Exact duplicate audio files detected. Review "
            "artifacts/baseline/possible_exact_duplicates.csv and resolve them "
            "before splitting to avoid leakage."
        )

    test_val_fraction = TEST_FRACTION + VALIDATION_FRACTION
    train_df, temporary_df = train_test_split(
        df,
        test_size=test_val_fraction,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    test_share_of_temp = TEST_FRACTION / test_val_fraction
    val_df, test_df = train_test_split(
        temporary_df,
        test_size=test_share_of_temp,
        random_state=RANDOM_SEED,
        stratify=temporary_df["label"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="validation")
    test_df = test_df.assign(split="test")
    result = pd.concat([train_df, val_df, test_df], ignore_index=True)
    result = result.sort_values(["split", "label", "path"]).reset_index(drop=True)

    # Guard against file-level overlap.
    path_sets = {
        name: set(result.loc[result.split == name, "path"])
        for name in ("train", "validation", "test")
    }
    assert path_sets["train"].isdisjoint(path_sets["validation"])
    assert path_sets["train"].isdisjoint(path_sets["test"])
    assert path_sets["validation"].isdisjoint(path_sets["test"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SPLIT_PATH, index=False)
    print(f"Split manifest: {SPLIT_PATH}")
    print("\nCounts by split and class:")
    print(pd.crosstab(result["split"], result["label"]).reindex(
        index=["train", "validation", "test"], columns=CLASSES, fill_value=0
    ).to_string())
    print(
        "\nLIMITATION: this is a stratified file-level split. If multiple WAVs "
        "come from the same chicken, session, or source recording, use source "
        "metadata and a group-based split before making strong generalization claims."
    )


if __name__ == "__main__":
    main()
