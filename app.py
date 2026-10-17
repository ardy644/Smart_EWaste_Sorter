"""
app.py — Smart E-Waste & Recyclable Vision Sorter
===================================================
A sleek dark-mode Streamlit dashboard with 4 tabs:
  1. Live CNN Sorter & Grad-CAM XAI
  2. Exploratory Data Analysis
  3. Model Evaluation & Comparison
  4. Batch Inspection & Sorting Manifest
"""

from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from plotly.subplots import make_subplots

from model_utils import (
    BASE_DIR,
    CLASSES,
    METRICS,
    NUM_CLASSES,
    SAMPLE_IMG_DIR,
    BaselineCNN,
    count_parameters,
    generate_gradcam_overlay,
    load_baseline,
    load_efficientnet,
    predict,
    preprocess_image,
)

# ═══════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="♻️ Smart E-Waste Vision Sorter",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
/* ---------- global dark overrides ---------- */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"]         { background: #161b22; }
h1, h2, h3, h4, p, span, label, div { color: #e6edf3 !important; }

/* KPI metric cards */
.kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,.35);
}
.kpi-card .kpi-value {
    font-size: 2.0rem;
    font-weight: 700;
    margin: 4px 0;
}
.kpi-card .kpi-label {
    font-size: 0.85rem;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Disposal card */
.disposal-card {
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,.35);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Cached model loaders
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = "cpu"


@st.cache_resource(show_spinner="Loading EfficientNetV2-S …")
def _load_efficientnet():
    return load_efficientnet(DEVICE)


@st.cache_resource(show_spinner="Loading Baseline CNN …")
def _load_baseline():
    return load_baseline(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
META = METRICS["class_metadata"]
SAMPLE_MANIFEST = METRICS["sample_images"]

BIN_EMOJI = {
    "Red / Specialized E-Waste Drop-off": "🔴",
    "Blue Dry Recyclable Bin": "🔵",
    "Green Compost Bin": "🟢",
}

CATEGORY_COLOURS = {
    "Hazardous E-Waste": "#e74c3c",
    "Dry Recyclable / Household": "#3498db",
    "Wet / Biodegradable Waste": "#2ecc71",
}


def _kpi_html(value: str, label: str, colour: str = "#58a6ff") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{colour} !important;">{value}</div>
    </div>"""


def _disposal_card(cls: str) -> str:
    m = META.get(cls, {})
    cat = m.get("category", "Unknown")
    bg = CATEGORY_COLOURS.get(cat, "#30363d")
    emoji = BIN_EMOJI.get(m.get("bin_color", ""), "🗑️")
    return f"""
    <div class="disposal-card" style="border-left: 5px solid {bg}; background: #161b22;">
        <h4 style="margin:0 0 6px 0;">{emoji} Disposal Card — {cls}</h4>
        <table style="width:100%; font-size:0.92rem;">
            <tr><td style="color:#8b949e !important; width:130px;">Category</td>
                <td><b style="color:{bg} !important;">{cat}</b></td></tr>
            <tr><td style="color:#8b949e !important;">Bin</td>
                <td>{m.get("bin_color", "N/A")}</td></tr>
            <tr><td style="color:#8b949e !important;">Hazard</td>
                <td>{m.get("hazard_level", "N/A")}</td></tr>
            <tr><td style="color:#8b949e !important;">Handling</td>
                <td style="font-style:italic;">{m.get("handling_tip", "N/A")}</td></tr>
        </table>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ♻️ Vision Sorter")
    st.markdown("---")
    st.markdown(
        f"**Dataset**: {METRICS['total_raw_images']:,} raw → "
        f"{METRICS['total_balanced_images']:,} balanced"
    )
    st.markdown(f"**Classes**: {NUM_CLASSES}")
    st.markdown(f"**Device**: `{DEVICE.upper()}`")
    st.markdown("---")
    st.caption("Built with PyTorch · Streamlit · Plotly")

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Live Sorter & Grad-CAM",
    "📊 Exploratory Data Analysis",
    "📈 Model Evaluation",
    "📦 Batch Inspection",
])


# ───────────────────── TAB 1 — Live Sorter & Grad-CAM ────────────────────
with tab1:
    st.header("🔬 Live CNN Sorter & Grad-CAM Explainable AI")

    col_input, col_opts = st.columns([2, 1])
    with col_input:
        img_source = st.radio(
            "Image source",
            ["Upload an image", "Select a sample image"],
            horizontal=True,
        )
    with col_opts:
        model_choice = st.selectbox(
            "Model",
            ["EfficientNetV2-S (Transfer Learning)", "Baseline 4-Layer CNN", "Both (Side-by-Side)"],
        )

    # Acquire image ---
    pil_image: Image.Image | None = None
    if img_source == "Upload an image":
        uploaded = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            pil_image = Image.open(uploaded)
    else:
        sample_names = [s["filename"] for s in SAMPLE_MANIFEST]
        chosen = st.selectbox("Pick a sample", sample_names)
        if chosen:
            pil_image = Image.open(SAMPLE_IMG_DIR / chosen)

    if pil_image is not None:
        input_tensor = preprocess_image(pil_image)

        # Determine which models to run ---
        run_models: list[tuple[str, object]] = []
        if model_choice == "Both (Side-by-Side)":
            run_models = [
                ("EfficientNetV2-S", _load_efficientnet()),
                ("Baseline 4-Layer CNN", _load_baseline()),
            ]
        elif "EfficientNet" in model_choice:
            run_models = [("EfficientNetV2-S", _load_efficientnet())]
        else:
            run_models = [("Baseline 4-Layer CNN", _load_baseline())]

        cols = st.columns(len(run_models))
        for i, (mname, model) in enumerate(run_models):
            with cols[i]:
                st.subheader(mname)
                pred_cls, conf, probs, latency = predict(model, input_tensor, DEVICE)

                # KPI row
                k1, k2, k3 = st.columns(3)
                k1.markdown(_kpi_html(pred_cls, "Prediction", "#58a6ff"), unsafe_allow_html=True)
                k2.markdown(_kpi_html(f"{conf:.1f}%", "Confidence", "#3fb950"), unsafe_allow_html=True)
                k3.markdown(_kpi_html(f"{latency:.0f} ms", "Latency", "#d29922"), unsafe_allow_html=True)

                # Disposal card
                st.markdown(_disposal_card(pred_cls), unsafe_allow_html=True)

                # Top-5 bar chart
                top5_idx = np.argsort(probs)[::-1][:5]
                top5_cls = [CLASSES[j] for j in top5_idx]
                top5_prb = [float(probs[j]) * 100 for j in top5_idx]
                colours = ["#58a6ff" if c == pred_cls else "#30363d" for c in top5_cls]
                fig_bar = go.Figure(go.Bar(
                    x=top5_prb[::-1],
                    y=top5_cls[::-1],
                    orientation="h",
                    marker_color=colours[::-1],
                    text=[f"{v:.1f}%" for v in top5_prb[::-1]],
                    textposition="auto",
                ))
                fig_bar.update_layout(
                    title="Top-5 Predictions",
                    xaxis_title="Confidence %",
                    yaxis_title="",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=30),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Grad-CAM
                st.markdown("#### 🔥 Grad-CAM Activation Map")
                orig_rgb, heatmap_rgb, overlay_rgb = generate_gradcam_overlay(
                    model, mname, input_tensor, pil_image,
                )
                gc1, gc2, gc3 = st.columns(3)
                gc1.image(orig_rgb, caption="Original", use_container_width=True)
                gc2.image(heatmap_rgb, caption="Raw Activation", use_container_width=True)
                gc3.image(overlay_rgb, caption="Grad-CAM Overlay", use_container_width=True)
    else:
        st.info("👆 Upload or select an image to begin inference.")


# ───────────────────── TAB 2 — Exploratory Data Analysis ─────────────────
with tab2:
    st.header("📊 Exploratory Data Analysis")

    # --- Raw vs Balanced Distribution ---
    st.subheader("Class Distribution: Raw vs. Balanced")
    raw_dist = METRICS["raw_class_distribution"]
    bal_dist = METRICS["balanced_class_distribution"]
    dist_df = pd.DataFrame({
        "Class": list(raw_dist.keys()),
        "Raw": list(raw_dist.values()),
        "Balanced": [bal_dist.get(c, 0) for c in raw_dist.keys()],
    }).sort_values("Raw", ascending=False)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=dist_df["Class"], y=dist_df["Raw"],
        name="Raw", marker_color="#e74c3c", opacity=0.8,
    ))
    fig_dist.add_trace(go.Bar(
        x=dist_df["Class"], y=dist_df["Balanced"],
        name="Balanced", marker_color="#3fb950", opacity=0.8,
    ))
    fig_dist.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=30, b=100),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Super-Category Donut ---
    st.subheader("Super-Category Breakdown")
    cat_counts: dict[str, int] = {}
    for cls in CLASSES:
        cat = META.get(cls, {}).get("category", "Other")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_df = pd.DataFrame({"Category": list(cat_counts.keys()), "Count": list(cat_counts.values())})
    cat_colours = [CATEGORY_COLOURS.get(c, "#888") for c in cat_df["Category"]]
    fig_donut = go.Figure(go.Pie(
        labels=cat_df["Category"],
        values=cat_df["Count"],
        hole=0.5,
        marker=dict(colors=cat_colours),
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # --- Sample Image Grid ---
    st.subheader("📷 Sample Image Gallery")
    unique_labels = sorted(set(s["true_label"] for s in SAMPLE_MANIFEST))
    filter_label = st.multiselect("Filter by class", unique_labels, default=unique_labels)

    filtered = [s for s in SAMPLE_MANIFEST if s["true_label"] in filter_label]
    grid_cols = st.columns(4)
    for idx, sample in enumerate(filtered):
        with grid_cols[idx % 4]:
            img_path = SAMPLE_IMG_DIR / sample["filename"]
            if img_path.exists():
                meta = META.get(sample["true_label"], {})
                cat = meta.get("category", "")
                colour = CATEGORY_COLOURS.get(cat, "#888")
                st.image(
                    str(img_path),
                    caption=f"{sample['true_label']}",
                    use_container_width=True,
                )
                st.markdown(
                    f"<small style='color:{colour} !important;'>{cat}</small>",
                    unsafe_allow_html=True,
                )


# ──────────────── TAB 3 — Model Evaluation & Comparison ──────────────────
with tab3:
    st.header("📈 Model Evaluation & Comparison")
    models_data = METRICS["models"]

    # --- KPI cards ---
    st.subheader("Performance Summary")
    eff_data = models_data["EfficientNetV2_S"]
    bas_data = models_data["Baseline_CNN"]

    eff_params = count_parameters(_load_efficientnet())
    bas_params = count_parameters(_load_baseline())

    k_cols = st.columns(6)
    items = [
        ("Baseline CNN", f"{bas_data['best_val_accuracy']*100:.1f}%", "Val Accuracy", "#e74c3c"),
        ("Baseline CNN", f"{bas_data['macro_f1']:.3f}", "Macro F1", "#e74c3c"),
        ("Baseline CNN", f"{bas_params/1e6:.2f}M", "Parameters", "#e74c3c"),
        ("EfficientNetV2-S", f"{eff_data['best_val_accuracy']*100:.1f}%", "Val Accuracy", "#3fb950"),
        ("EfficientNetV2-S", f"{eff_data['macro_f1']:.3f}", "Macro F1", "#3fb950"),
        ("EfficientNetV2-S", f"{eff_params/1e6:.1f}M", "Parameters", "#3fb950"),
    ]
    for i, (mname, val, lbl, clr) in enumerate(items):
        k_cols[i].markdown(_kpi_html(val, f"{mname}<br>{lbl}", clr), unsafe_allow_html=True)

    st.markdown("---")

    # --- Learning Curves ---
    st.subheader("Learning Curves")
    lc1, lc2 = st.columns(2)

    for col, metric_key, title in [
        (lc1, "train_acc", "Accuracy"),
        (lc2, "train_loss", "Loss"),
    ]:
        with col:
            fig_lc = make_subplots()
            for mname_key, mname_label, dash in [
                ("Baseline_CNN", "Baseline", "dot"),
                ("EfficientNetV2_S", "EfficientNet", "solid"),
            ]:
                hist = models_data[mname_key]["history"]
                train_key = metric_key
                val_key = metric_key.replace("train_", "val_")
                epochs = list(range(1, len(hist[train_key]) + 1))

                fig_lc.add_trace(go.Scatter(
                    x=epochs, y=hist[train_key],
                    name=f"{mname_label} Train",
                    line=dict(dash=dash, width=2),
                    mode="lines+markers",
                ))
                fig_lc.add_trace(go.Scatter(
                    x=epochs, y=hist[val_key],
                    name=f"{mname_label} Val",
                    line=dict(dash=dash, width=2.5),
                    mode="lines+markers",
                ))

            fig_lc.update_layout(
                title=f"Training & Validation {title}",
                xaxis_title="Epoch",
                yaxis_title=title,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                legend=dict(font_size=10),
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown("---")

    # --- Confusion Matrices ---
    st.subheader("Confusion Matrices (21 × 21)")
    cm_tab_b, cm_tab_e = st.tabs(["Baseline CNN", "EfficientNetV2-S"])
    for cm_tab, mkey in [(cm_tab_b, "Baseline_CNN"), (cm_tab_e, "EfficientNetV2_S")]:
        with cm_tab:
            cm = np.array(models_data[mkey]["confusion_matrix"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=CLASSES,
                y=CLASSES,
                colorscale="Blues" if "Efficient" in mkey else "Reds",
                text=cm,
                texttemplate="%{text}",
                textfont_size=9,
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="True",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=600,
                xaxis=dict(tickangle=-45, tickfont_size=9),
                yaxis=dict(tickfont_size=9, autorange="reversed"),
                margin=dict(l=80, r=20, t=30, b=100),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # --- Per-class Precision / Recall / F1 ---
    st.subheader("Per-Class Metrics")
    sel_model_cls = st.selectbox("Model", list(models_data.keys()), key="cls_report_model")
    report = models_data[sel_model_cls]["classification_report"]
    rows = []
    for cls in CLASSES:
        if cls in report:
            r = report[cls]
            rows.append({
                "Class": cls,
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1-Score": round(r["f1-score"], 4),
                "Support": int(r["support"]),
            })
    cls_df = pd.DataFrame(rows)
    # F1-Score bar chart (visual replacement for background_gradient)
    fig_f1 = go.Figure(go.Bar(
        x=cls_df["F1-Score"],
        y=cls_df["Class"],
        orientation="h",
        marker_color=cls_df["F1-Score"],
        marker_colorscale="RdYlGn",
        marker_cmin=0,
        marker_cmax=1,
        text=[f"{v:.3f}" for v in cls_df["F1-Score"]],
        textposition="auto",
    ))
    fig_f1.update_layout(
        title="Per-Class F1-Score",
        xaxis_title="F1-Score",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550,
        margin=dict(l=120, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.dataframe(cls_df, use_container_width=True, height=500)


# ──────────────── TAB 4 — Batch Inspection ───────────────────────────────
with tab4:
    st.header("📦 Batch Inspection & Automated Sorting Manifest")

    batch_model_name = st.selectbox(
        "Inference model",
        ["EfficientNetV2-S", "Baseline 4-Layer CNN"],
        key="batch_model",
    )
    if "EfficientNet" in batch_model_name:
        batch_model = _load_efficientnet()
    else:
        batch_model = _load_baseline()

    col_up, col_demo = st.columns(2)
    with col_up:
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
    with col_demo:
        st.markdown("**— or —**")
        run_demo = st.button("🎲 Run Batch Demo on 10 Random Samples")

    # Collect images ---
    images: list[tuple[str, Image.Image]] = []
    if batch_files:
        for f in batch_files:
            images.append((f.name, Image.open(f)))
    elif run_demo:
        chosen_samples = random.sample(SAMPLE_MANIFEST, min(10, len(SAMPLE_MANIFEST)))
        for s in chosen_samples:
            p = SAMPLE_IMG_DIR / s["filename"]
            if p.exists():
                images.append((s["filename"], Image.open(p)))

    if images:
        progress = st.progress(0, text="Running batch inference …")
        results = []
        for idx, (fname, img) in enumerate(images):
            tensor = preprocess_image(img)
            pred_cls, conf, probs, latency = predict(batch_model, tensor, DEVICE)
            m = META.get(pred_cls, {})
            results.append({
                "Filename": fname,
                "Predicted Class": pred_cls,
                "Confidence (%)": round(conf, 2),
                "Category": m.get("category", ""),
                "Hazard Level": m.get("hazard_level", ""),
                "Assigned Bin": m.get("bin_color", ""),
            })
            progress.progress((idx + 1) / len(images), text=f"Processed {idx + 1}/{len(images)}")

        progress.empty()
        result_df = pd.DataFrame(results)

        st.subheader("Sorting Manifest")
        st.dataframe(result_df, use_container_width=True, height=400)

        # Download CSV
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV Manifest",
            data=csv_bytes,
            file_name="sorting_manifest.csv",
            mime="text/csv",
        )

        # Thumbnail gallery
        st.subheader("Thumbnails")
        thumb_cols = st.columns(5)
        for idx, (fname, img) in enumerate(images):
            with thumb_cols[idx % 5]:
                st.image(img, caption=result_df.iloc[idx]["Predicted Class"], use_container_width=True)
    else:
        st.info("Upload images or click the demo button to run batch inference.")
