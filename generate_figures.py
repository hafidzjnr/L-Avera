"""
generate_figures.py
Menghasilkan seluruh gambar yang belum tersedia untuk laporan L-AVERA.

Gambar 3 (confusion matrix) TIDAK dibuat ulang karena sudah ada.
Gambar 4 (akurasi & macro-f1) sudah ada terpisah → dibuat versi gabungan.

Output disimpan di: results/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# ─── Konfigurasi Path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data" / "PoultryVocalization" / "Chicken_Audio_Dataset"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
EFFICIENCY_CSV = RESULTS / "effisiensi" / "efficiency_table.csv"
COMPARISON_CSV = RESULTS / "matriks" / "comparison_table.csv"

# Warna palet konsisten
COLORS = {
    "Baseline CNN": "#2E86AB",
    "Compact CNN":  "#A23B72",
    "DS-CNN":       "#F18F01",
    "TC-ResNet8":   "#C73E1D",
    "INT8":         "#3B1F2B",
}
CLASS_COLORS = {
    "Healthy":   "#2E86AB",
    "Noise":     "#F18F01",
    "Unhealthy": "#C73E1D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 200,
})


# ═══════════════════════════════════════════════════════════════════════
# Gambar 1: Visualisasi Fitur MFCC
# ═══════════════════════════════════════════════════════════════════════
def generate_mfcc_visualization():
    """Visualisasi MFCC dari satu sampel per kelas."""
    try:
        import librosa
        import librosa.display
    except ImportError:
        print("[SKIP] Gambar 1: librosa tidak tersedia. Install dulu: pip install librosa")
        return

    classes = ["Healthy", "Noise", "Unhealthy"]
    sample_rate = 22050
    n_mfcc = 40
    max_frames = 200

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    fig.suptitle("Visualisasi Fitur MFCC per Kelas", fontsize=14, fontweight="bold")

    for ax, cls in zip(axes, classes):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"[SKIP] Gambar 1: Folder {cls_dir} tidak ditemukan.")
            return

        wav_files = sorted(cls_dir.glob("*.wav"))
        if not wav_files:
            print(f"[SKIP] Gambar 1: Tidak ada file WAV di {cls_dir}.")
            return

        audio, _ = librosa.load(wav_files[0], sr=sample_rate, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :max_frames]

        img = librosa.display.specshow(
            mfcc, sr=sample_rate, x_axis="frames", y_axis=None, ax=ax, cmap="magma"
        )
        ax.set_title(cls, color=CLASS_COLORS[cls], fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Koefisien MFCC" if cls == "Healthy" else "")

    cbar = fig.colorbar(img, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Amplitudo MFCC")

    out = RESULTS / "gambar1_mfcc_visualization.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gambar 1 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Gambar 2: Distribusi Kelas Dataset
# ═══════════════════════════════════════════════════════════════════════
def generate_class_distribution():
    """Pie chart dan bar chart distribusi kelas."""
    class_counts = {}
    for cls in ["Healthy", "Noise", "Unhealthy"]:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"[SKIP] Gambar 2: Folder {cls_dir} tidak ditemukan.")
            return
        class_counts[cls] = len(list(cls_dir.glob("*.wav")))

    if sum(class_counts.values()) == 0:
        print("[SKIP] Gambar 2: Tidak ada file WAV di dataset.")
        return

    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS[c] for c in labels]
    total = sum(counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.suptitle("Distribusi Kelas Dataset Poultry Vocalization", fontsize=14, fontweight="bold")

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax1.set_title(f"Proporsi (Total: {total})")

    # Bar chart
    bars = ax2.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    ax2.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Jumlah Sampel")
    ax2.set_title("Jumlah per Kelas")
    ax2.set_ylim(0, max(counts) * 1.18)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = RESULTS / "gambar2_distribusi_kelas.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gambar 2 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Gambar 3: Confusion Matrix Gabungan (4 Model)
# ═══════════════════════════════════════════════════════════════════════
def generate_confusion_matrix_combined():
    """Menggabungkan confusion matrix dari 4 model ke dalam satu gambar (grid 2x2)."""
    import seaborn as sns

    classes = ["Healthy", "Noise", "Unhealthy"]

    # Data confusion matrix hasil evaluasi held-out test set (52 sampel)
    # Nilai diverifikasi dari evaluasi masing-masing model
    cms = {
        "Baseline CNN": {
            "matrix": np.array([
                [21,  0,  0],
                [ 1, 11,  1],
                [ 0,  0, 18]
            ]),
            "acc": "96.15%",
            "f1": "95.55%"
        },
        "Compact CNN": {
            "matrix": np.array([
                [21,  0,  0],
                [ 5,  7,  1],
                [ 0,  0, 18]
            ]),
            "acc": "88.46%",
            "f1": "85.55%"
        },
        "DS-CNN": {
            "matrix": np.array([
                [21,  0,  0],
                [13,  0,  0],
                [ 5,  2, 11]
            ]),
            "acc": "61.54%",
            "f1": "48.62%"
        },
        "TC-ResNet8": {
            "matrix": np.array([
                [21,  0,  0],
                [ 2, 10,  1],
                [ 0,  0, 18]
            ]),
            "acc": "94.23%",
            "f1": "93.24%"
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    fig.suptitle(
        "Perbandingan Confusion Matrix Antar Model (Held-out Test Set, N=52)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    models = list(cms.keys())
    sublabels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, (ax, model_name, sublabel) in enumerate(zip(axes.flat, models, sublabels)):
        data = cms[model_name]
        cm = data["matrix"]
        acc = data["acc"]
        f1 = data["f1"]

        title_suffix = " (Float32 & INT8)" if model_name == "TC-ResNet8" else ""

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            vmin=0,
            vmax=22,
            cbar=(idx % 2 == 1),
            cbar_kws={"label": "Jumlah Sampel"} if idx == 1 else None,
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            linewidths=0.8,
            linecolor="white",
        )

        ax.set_title(
            f"{sublabel} {model_name}{title_suffix}\nAkurasi: {acc} | Macro-F1: {f1}",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Prediksi", fontsize=10, fontweight="bold")
        ax.set_ylabel("Aktual", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=10)

    fig.text(
        0.5,
        0.012,
        "*Catatan: Evaluasi TC-ResNet8 INT8 menghasilkan matriks konfusi yang identik (100% konsisten) dengan versi Float32.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = RESULTS / "gambar3_confusion_matrix_gabungan.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Gambar 3 tersimpan: {out}")



# ═══════════════════════════════════════════════════════════════════════
# Gambar 4: Grafik Perbandingan Akurasi DAN Macro-F1 (gabungan)
# ═══════════════════════════════════════════════════════════════════════
def generate_accuracy_f1_comparison():
    """Grouped bar chart: akurasi dan Macro-F1 berdampingan."""
    if not COMPARISON_CSV.exists():
        print(f"[SKIP] Gambar 4: {COMPARISON_CSV} tidak ditemukan.")
        return

    df = pd.read_csv(COMPARISON_CSV)
    models = df["Model"].tolist()
    accuracy = df["accuracy"].tolist()
    macro_f1 = df["macro_f1"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy",
                   color="#2E86AB", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1",
                   color="#C73E1D", edgecolor="white", linewidth=0.8)

    ax.bar_label(bars1, fmt="{:.2%}", padding=3, fontsize=8.5)
    ax.bar_label(bars2, fmt="{:.2%}", padding=3, fontsize=8.5)

    ax.set_ylabel("Skor")
    ax.set_title("Perbandingan Akurasi dan Macro-F1 Antar Arsitektur")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.13)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = RESULTS / "gambar4_accuracy_f1_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gambar 4 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Gambar 5: Perbandingan Jumlah Parameter dan MACCs
# ═══════════════════════════════════════════════════════════════════════
def generate_params_maccs_comparison():
    """Grouped bar chart: parameter dan MACCs per model."""
    if not EFFICIENCY_CSV.exists():
        print(f"[SKIP] Gambar 5: {EFFICIENCY_CSV} tidak ditemukan.")
        return

    df = pd.read_csv(EFFICIENCY_CSV)

    # Filter hanya model float32 (bukan INT8 untuk parameter)
    df_params = df[df["Total Parameters"] != "N/A"].copy()
    df_params["Total Parameters"] = df_params["Total Parameters"].astype(float)

    models_params = df_params["Model"].tolist()
    params = df_params["Total Parameters"].tolist()

    # MACCs semua model
    models_maccs = df["Model"].tolist()
    maccs = df["MACCs"].astype(float).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Perbandingan Jumlah Parameter dan MACCs", fontsize=14, fontweight="bold")

    # Parameter
    colors_p = [COLORS.get(m, "#666") for m in models_params]
    bars1 = ax1.barh(models_params, [p / 1e3 for p in params], color=colors_p,
                     edgecolor="white", linewidth=0.8, height=0.5)
    ax1.bar_label(bars1, labels=[f"{p:,.0f}" for p in params], padding=4, fontsize=9)
    ax1.set_xlabel("Parameter (×1.000)")
    ax1.set_title("Jumlah Parameter")
    ax1.invert_yaxis()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    # MACCs
    colors_m = [COLORS.get(m, "#666") for m in models_maccs]
    bars2 = ax2.barh(models_maccs, [m / 1e6 for m in maccs], color=colors_m,
                     edgecolor="white", linewidth=0.8, height=0.5)
    ax2.bar_label(bars2, labels=[f"{m:,.0f}" for m in maccs], padding=4, fontsize=9)
    ax2.set_xlabel("MACCs (×1.000.000)")
    ax2.set_title("Beban Komputasi (MACCs)")
    ax2.invert_yaxis()
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = RESULTS / "gambar5_params_maccs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gambar 5 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Gambar 6: Perbandingan Ukuran Model Float32 vs INT8
# ═══════════════════════════════════════════════════════════════════════
def generate_model_size_comparison():
    """Bar chart: ukuran model Float32 vs INT8 (TC-ResNet8)."""
    if not EFFICIENCY_CSV.exists():
        print(f"[SKIP] Gambar 6: {EFFICIENCY_CSV} tidak ditemukan.")
        return

    df = pd.read_csv(EFFICIENCY_CSV)

    # Semua model dan ukurannya
    models = df["Model"].tolist()
    sizes_kb = df["File Size (KB)"].astype(float).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.5, 1]})
    fig.suptitle("Perbandingan Ukuran Model", fontsize=14, fontweight="bold")

    # Panel kiri: semua model
    colors_all = [COLORS.get(m, "#666") for m in models]
    bars_all = ax1.bar(models, sizes_kb, color=colors_all,
                       edgecolor="white", linewidth=0.8, width=0.55)
    for bar, sz in zip(bars_all, sizes_kb):
        if sz > 1000:
            label = f"{sz/1024:.1f} MB"
        else:
            label = f"{sz:.1f} KB"
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes_kb)*0.02,
                 label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Ukuran (KB)")
    ax1.set_title("Ukuran Berkas Seluruh Model")
    ax1.set_ylim(0, max(sizes_kb) * 1.2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel kanan: Float32 vs INT8 (TC-ResNet8 saja)
    tcresnet_row = df[df["Model"] == "TC-ResNet8"]
    int8_row = df[df["Model"] == "INT8"]
    if not tcresnet_row.empty and not int8_row.empty:
        fp32_kb = float(tcresnet_row["File Size (KB)"].values[0])
        int8_kb = float(int8_row["File Size (KB)"].values[0])
        reduction_pct = (1 - int8_kb / fp32_kb) * 100

        bar_labels = ["Float32\n(TC-ResNet8)", "INT8\n(TC-ResNet8)"]
        bar_vals = [fp32_kb, int8_kb]
        bar_colors = [COLORS["TC-ResNet8"], COLORS["INT8"]]

        bars_cmp = ax2.bar(bar_labels, bar_vals, color=bar_colors,
                           edgecolor="white", linewidth=0.8, width=0.5)
        for bar, sz in zip(bars_cmp, bar_vals):
            label = f"{sz:.1f} KB"
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + fp32_kb*0.03,
                     label, ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.annotate(
            f"Reduksi {reduction_pct:.1f}%",
            xy=(1, int8_kb), xytext=(0.5, fp32_kb * 0.65),
            fontsize=11, fontweight="bold", color="#C73E1D",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#C73E1D", lw=1.5),
        )
        ax2.set_ylabel("Ukuran (KB)")
        ax2.set_title("Float32 vs INT8")
        ax2.set_ylim(0, fp32_kb * 1.25)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = RESULTS / "gambar6_model_size_float32_int8.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gambar 6 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Gambar 7: Diagram Alur Penerapan Model pada Perangkat Edge
# ═══════════════════════════════════════════════════════════════════════
def generate_edge_pipeline_diagram():
    """Diagram alur: Audio → MFCC → TC-ResNet8 INT8 → Label."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # Definisi kotak alur
    steps = [
        {"x": 0.3,  "label": "Input Audio\n(WAV)", "color": "#2E86AB", "icon": "[1]"},
        {"x": 3.0,  "label": "Preprocessing\nMFCC 40 x 200", "color": "#A23B72", "icon": "[2]"},
        {"x": 6.0,  "label": "TC-ResNet8\nINT8", "color": "#C73E1D", "icon": "[3]"},
        {"x": 9.0,  "label": "Inferensi\nESP32-S3", "color": "#F18F01", "icon": "[4]"},
        {"x": 11.7, "label": "Label\nKlasifikasi", "color": "#1B998B", "icon": "[5]"},
    ]

    box_w = 2.2
    box_h = 1.6
    y_center = 1.75

    for step in steps:
        x = step["x"]
        y = y_center - box_h / 2

        fancy = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=step["color"], edgecolor="white",
            linewidth=2, alpha=0.9
        )
        ax.add_patch(fancy)

        ax.text(x + box_w / 2, y + box_h * 0.65, step["icon"],
                ha="center", va="center", fontsize=12, color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.3))
        ax.text(x + box_w / 2, y + box_h * 0.28, step["label"],
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold", linespacing=1.3)

    # Panah antar kotak
    arrow_style = dict(arrowstyle="-|>", color="#555", lw=2, mutation_scale=18)
    for i in range(len(steps) - 1):
        x_start = steps[i]["x"] + box_w + 0.05
        x_end = steps[i + 1]["x"] - 0.05
        ax.annotate(
            "", xy=(x_end, y_center), xytext=(x_start, y_center),
            arrowprops=arrow_style,
        )

    ax.set_title("Diagram Alur Penerapan Model pada Perangkat Edge",
                 fontsize=13, fontweight="bold", pad=15)

    out = RESULTS / "gambar7_edge_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Gambar 7 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Gambar 8: Diagram Arsitektur Sistem L-AVERA
# ═══════════════════════════════════════════════════════════════════════
def generate_system_architecture():
    """Diagram arsitektur lengkap sistem L-AVERA."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel="", color="#2E86AB", icon=""):
        fancy = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.92
        )
        ax.add_patch(fancy)
        if icon:
            ax.text(x + w / 2, y + h * 0.75, icon,
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.25))
        ax.text(x + w / 2, y + h * 0.42, label,
                ha="center", va="center", fontsize=9.5, color="white",
                fontweight="bold", linespacing=1.3)
        if sublabel:
            ax.text(x + w / 2, y + h * 0.12, sublabel,
                    ha="center", va="center", fontsize=7.5, color="#ffffffcc",
                    style="italic", linespacing=1.2)

    def draw_arrow(x1, y1, x2, y2, label="", color="#555"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16),
        )
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=color, alpha=0.85))

    # ─── Layer 1: Perekaman Suara ───
    draw_box(0.5, 8.0, 3.0, 1.5,
             "Perekaman Suara", "3x INMP441 MEMS Mic",
             color="#2E86AB", icon="MIC")

    # ─── Layer 2: Pemrosesan Audio (ESP32-S3) ───
    draw_box(0.5, 5.5, 3.0, 1.5,
             "Pemrosesan Audio", "Mono - 22.050 Hz\nMFCC 40 - 200 frame",
             color="#A23B72", icon="DSP")

    # ─── Layer 3: Inferensi ───
    draw_box(5.0, 5.5, 3.5, 1.5,
             "Inferensi\nTC-ResNet8 INT8", "ESP32-S3 - TFLite Micro\n~95 KB - ~32 KB Arena",
             color="#C73E1D", icon="ML")

    # ─── Layer 3b: Micro-SD ───
    draw_box(10.0, 5.5, 3.0, 1.5,
             "Micro-SD", "Log lokal saat\njaringan terputus",
             color="#6C757D", icon="SD")

    # ─── Layer 4: Wi-Fi / HTTP POST ───
    draw_box(5.0, 3.0, 3.5, 1.5,
             "Wi-Fi / HTTP POST", "Pengiriman hasil\nklasifikasi",
             color="#F18F01", icon="NET")

    # ─── Layer 5: Server & Dashboard ───
    draw_box(5.0, 0.5, 3.5, 1.5,
             "Server & Dashboard", "Kelas prediksi, riwayat\nzona pengamatan",
             color="#1B998B", icon="WEB")

    # ─── ESP32-S3 Box (background) ───
    esp_rect = mpatches.FancyBboxPatch(
        (0.2, 5.2), 13.2, 2.1,
        boxstyle="round,pad=0.2",
        facecolor="none", edgecolor="#C73E1D",
        linewidth=2, linestyle="--", alpha=0.6
    )
    ax.add_patch(esp_rect)
    ax.text(13.2, 7.15, "ESP32-S3", fontsize=10, fontweight="bold",
            color="#C73E1D", ha="right", va="bottom", alpha=0.7)

    # ─── Panah ───
    # Perekaman → Pemrosesan
    draw_arrow(2.0, 8.0, 2.0, 7.05, label="Audio RAW")
    # Pemrosesan → Inferensi
    draw_arrow(3.5, 6.25, 5.0, 6.25, label="MFCC Tensor")
    # Inferensi → Micro-SD
    draw_arrow(8.5, 6.25, 10.0, 6.25, label="Fallback Log")
    # Inferensi → Wi-Fi
    draw_arrow(6.75, 5.5, 6.75, 4.55, label="Hasil Prediksi")
    # Wi-Fi → Server
    draw_arrow(6.75, 3.0, 6.75, 2.05, label="HTTP POST")

    ax.set_title("Diagram Arsitektur Sistem L-AVERA",
                 fontsize=14, fontweight="bold", pad=20)

    out = RESULTS / "gambar8_arsitektur_sistem.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Gambar 8 tersimpan: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  L-AVERA — Generate Figures")
    print("=" * 60)

    # Gambar 1: MFCC Visualization
    print("\n[1/7] Gambar 1: Visualisasi MFCC...")
    generate_mfcc_visualization()

    # Gambar 2: Distribusi Kelas
    print("\n[2/7] Gambar 2: Distribusi kelas dataset...")
    generate_class_distribution()

    # Gambar 3: Confusion Matrix Gabungan (4 Model)
    print("\n[3/7] Gambar 3: Confusion matrix gabungan (4 model)...")
    generate_confusion_matrix_combined()

    # Gambar 4: Akurasi & Macro-F1 (gabungan)
    print("\n[4/7] Gambar 4: Perbandingan Akurasi dan Macro-F1...")
    generate_accuracy_f1_comparison()

    # Gambar 5: Parameter & MACCs
    print("\n[5/7] Gambar 5: Perbandingan Parameter dan MACCs...")
    generate_params_maccs_comparison()

    # Gambar 6: Ukuran Model Float32 vs INT8
    print("\n[6/7] Gambar 6: Perbandingan Ukuran Model...")
    generate_model_size_comparison()

    # Gambar 7: Diagram Alur Edge
    print("\n[7/7] Gambar 7: Diagram Alur Edge Pipeline...")
    generate_edge_pipeline_diagram()

    # Gambar 8: Arsitektur Sistem
    print("\n[8/7] Gambar 8: Arsitektur Sistem L-AVERA...")
    generate_system_architecture()

    print("\n" + "=" * 60)
    print("  Selesai! Cek folder: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
