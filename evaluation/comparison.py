# Pembuatan tabel komparasi dan visualisasi perbandingan seluruh eksperimen model
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import SCALAR_KEYS, load_all_available
from evaluation.efficiency import all_efficiency_reports

# Direktori penyimpanan hasil evaluasi perbandingan
RESULTS_METRICS_DIR = Path("results/matriks")
RESULTS_EFFICIENCY_DIR = Path("results/effisiensi")
RESULTS_PLOT_DIR = Path("results/plot")

# Label nama model untuk tampilan laporan
MODEL_LABELS = {
    "baseline_cnn": "Baseline CNN",
    "compact_cnn":  "Compact CNN",
    "ds_cnn":       "DS-CNN",
    "tc_resnet8":   "TC-ResNet8",
    "int8":         "INT8",
}


# --- Fungsi Pembuat Grafik Diagram Batang ---
def _bar_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color_palette: list[str] | None = None,
) -> None:
    colors = color_palette or ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        df["Model"], df[column],
        color=colors[: len(df)], edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(bars, fmt="{:.4f}", padding=3, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Grafik tersimpan: {output_path}")


def main() -> None:
    for d in (RESULTS_METRICS_DIR, RESULTS_EFFICIENCY_DIR, RESULTS_PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Rekapitulasi tabel performa metrik klasifikasi
    all_metrics = load_all_available()
    if not all_metrics:
        print("Belum ada berkas metrik. Jalankan eksperimen pelatihan terlebih dahulu.")
        return

    rows = []
    for key, m in all_metrics.items():
        row = {"Model": MODEL_LABELS.get(key, key)}
        for scalar in SCALAR_KEYS:
            row[scalar] = round(m.get(scalar, float("nan")), 6)
        rows.append(row)

    perf_df = pd.DataFrame(rows)
    perf_csv = RESULTS_METRICS_DIR / "comparison_table.csv"
    perf_df.to_csv(perf_csv, index=False)
    print(f"Tabel performa tersimpan: {perf_csv}")
    print("\nRekapitulasi Performa:")
    print(perf_df.to_string(index=False))

    # Rekapitulasi tabel efisiensi model (parameter dan ukuran file)
    eff_reports = all_efficiency_reports()
    if eff_reports:
        eff_rows = []
        for r in eff_reports:
            params = r.get("parameters") or {}
            eff_rows.append({
                "Model": MODEL_LABELS.get(r["model"], r["model"]),
                "Total Parameters": params.get("total", "N/A"),
                "Trainable Parameters": params.get("trainable", "N/A"),
                "File Size (KB)": r["size_kb"],
            })
        eff_df = pd.DataFrame(eff_rows)
        eff_csv = RESULTS_EFFICIENCY_DIR / "efficiency_table.csv"
        eff_df.to_csv(eff_csv, index=False)
        print(f"\nTabel efisiensi tersimpan: {eff_csv}")
        print("\nRekapitulasi Efisiensi:")
        print(eff_df.to_string(index=False))

    # Pembuatan visualisasi grafik perbandingan
    if "macro_f1" in perf_df.columns:
        _bar_chart(
            perf_df.dropna(subset=["macro_f1"]),
            column="macro_f1",
            title="Perbandingan Macro-F1 Antar Arsitektur",
            ylabel="Macro-F1",
            output_path=RESULTS_PLOT_DIR / "macro_f1_comparison.png",
        )
    if "accuracy" in perf_df.columns:
        _bar_chart(
            perf_df.dropna(subset=["accuracy"]),
            column="accuracy",
            title="Perbandingan Akurasi Antar Arsitektur",
            ylabel="Accuracy",
            output_path=RESULTS_PLOT_DIR / "accuracy_comparison.png",
        )
    if "unhealthy_recall" in perf_df.columns:
        _bar_chart(
            perf_df.dropna(subset=["unhealthy_recall"]),
            column="unhealthy_recall",
            title="Perbandingan Recall Kelas Unhealthy Antar Arsitektur",
            ylabel="Recall Unhealthy",
            output_path=RESULTS_PLOT_DIR / "unhealthy_recall_comparison.png",
        )


if __name__ == "__main__":
    main()
