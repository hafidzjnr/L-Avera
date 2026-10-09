# Pemuatan dan standardisasi berkas metrik evaluasi per-eksperimen
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Jalur berkas hasil pengujian setiap model
METRIC_FILES: dict[str, Path] = {
    "baseline_cnn": Path("artifacts/baseline/test_metrics.json"),
    "compact_cnn":  Path("artifacts/compact_cnn/test_metrics.json"),
    "ds_cnn":       Path("artifacts/ds_cnn/test_metrics.json"),
    "tc_resnet8":   Path("artifacts/tcresnet/test_metrics.json"),
    "int8":         Path("artifacts/int8/test_metrics_int8.json"),
}

# Daftar kunci metrik skalar untuk tabel rekapitulasi
SCALAR_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "unhealthy_precision",
    "unhealthy_recall",
    "unhealthy_f1",
)


# --- Pemuatan Berkas Metrik Tunggal ---
def load_metrics(model_key: str) -> dict[str, Any]:
    path = METRIC_FILES[model_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Berkas metrik tidak ditemukan untuk '{model_key}': {path}\n"
            "Jalankan skrip pelatihan model terkait terlebih dahulu."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# --- Pemuatan Seluruh Metrik yang Tersedia ---
def load_all_available() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for key, path in METRIC_FILES.items():
        if path.exists():
            results[key] = json.loads(path.read_text(encoding="utf-8"))
    return results


# --- Ekstraksi Metrik Skalar ---
def scalar_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: metrics[k] for k in SCALAR_KEYS if k in metrics}
