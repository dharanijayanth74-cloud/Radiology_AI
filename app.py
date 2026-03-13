"""
RadAI – Chest X-Ray Analysis Pipeline
======================================
Streamlit application integrating all analysis modules in the following
pipeline execution order:

  1. Collect patient input        (sidebar)
  2. Upload X-ray                 (file uploader)
  3. Preprocess image             (preprocess.preprocess_image)
  4. Run autoencoder anomaly det. (autoencoder.process_and_evaluate)
  5. Run DenseNet anatomical ana. (cnn_analyzer.run_inference + get_target_layer)
  6. Run artifact detection       (artifact_detector.detect_artifacts)
  7. Generate Grad-CAM heatmap    (heatmap.generate_gradcam)
  8. Display results              (metrics, overlays, patient profile, predictions)
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ── Module imports ────────────────────────────────────────────────────────────
import preprocess          # Step 3 – image preprocessing
import autoencoder         # Step 4 – autoencoder anomaly detection
import cnn_analyzer        # Step 5 – DenseNet anatomical analysis
import artifact_detector   # Step 6 – artifact detection
import heatmap             # Step 7 – Grad-CAM heatmap generation

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RadAI – Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 RadAI – Chest X-Ray Analysis")
st.markdown(
    "Upload a chest X-ray and the AI pipeline will detect anomalies, "
    "identify anatomical findings, flag imaging artifacts, and visualise "
    "Grad-CAM attention maps."
)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 – Collect patient input
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.header("🏥 Step 1 – Patient Data")
patient_id  = st.sidebar.text_input("Patient ID", placeholder="e.g. 12345")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
patient_sex = st.sidebar.selectbox("Sex", ["—", "Male", "Female", "Other"])
symptoms    = st.sidebar.text_area("Clinical Symptoms / Notes",
                                    placeholder="Describe presenting symptoms…")

st.sidebar.markdown("---")
st.sidebar.header("👁️ Visualisation Toggles")
show_original = st.sidebar.checkbox("Show Original Image",     value=True)
show_anomaly  = st.sidebar.checkbox("Overlay Anomaly Map",     value=True)
show_gradcam  = st.sidebar.checkbox("Overlay Grad-CAM Heatmap",value=True)
show_artifacts= st.sidebar.checkbox("Highlight Artifacts",     value=True)

st.sidebar.markdown("---")
gradcam_opacity = st.sidebar.slider("Grad-CAM Opacity", 0.1, 0.9, 0.5, 0.1)

# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading Autoencoder model…")
def load_ae():
    """Load the convolutional autoencoder (Step 4)."""
    return autoencoder.load_autoencoder()

@st.cache_resource(show_spinner="🔧 Loading DenseNet121 model…")
def load_cnn():
    """Load the modified DenseNet121 classifier (Step 5)."""
    return cnn_analyzer.build_densenet()

ae_model  = load_ae()
cnn_model = load_cnn()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 – Upload X-ray
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📤 Step 2 – Upload Chest X-Ray")
uploaded_file = st.file_uploader(
    "Accepted formats: PNG, JPG, JPEG, DICOM (.dcm)",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Upload a standard chest posterior-anterior (PA) radiograph.",
)

# ── Pipeline runs only once a file is uploaded ────────────────────────────────
if uploaded_file is not None:

    pipeline_steps = [
        "Step 3 – Preprocessing image…",
        "Step 4 – Autoencoder anomaly detection…",
        "Step 5 – DenseNet anatomical analysis…",
        "Step 6 – Artifact detection…",
        "Step 7 – Generating Grad-CAM heatmap…",
    ]
    progress_bar = st.progress(0, text="Starting pipeline…")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 – Preprocess image
    # ─────────────────────────────────────────────────────────────────────────
    progress_bar.progress(0, text=pipeline_steps[0])
    try:
        tensor_img, original_pil = preprocess.preprocess_image(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Preprocessing failed: {exc}")
        st.stop()

    # Build a uint8 RGB array for OpenCV-based display operations
    original_np = np.array(original_pil)
    if original_np.ndim == 2:                              # grayscale → RGB
        original_display = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
    elif original_np.shape[2] == 4:                        # RGBA → RGB
        original_display = cv2.cvtColor(original_np, cv2.COLOR_RGBA2RGB)
    else:
        original_display = original_np.copy()

    # Ensure uint8 (some normalised arrays may be float)
    if original_display.dtype != np.uint8:
        original_display = np.clip(original_display * 255, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 – Run autoencoder anomaly detection
    # ─────────────────────────────────────────────────────────────────────────
    progress_bar.progress(20, text=pipeline_steps[1])
    # Build a single-channel tensor expected by the autoencoder (shape: 1×1×H×W)
    tensor_gray = preprocess.to_tensor(
        preprocess.normalize_image(
            preprocess.resize_image(
                preprocess.convert_to_grayscale(
                    preprocess.load_image(uploaded_file)
                )
            )
        )
    ).unsqueeze(0)   # add batch dim → (1, 1, 224, 224)

    ae_reconstructed, anomaly_heatmap, ae_loss = autoencoder.process_and_evaluate(
        ae_model, tensor_gray
    )
    # Determine if the loss exceeds the default anomaly threshold
    is_anomalous = autoencoder.is_anomalous(anomaly_heatmap,
                                             autoencoder.DEFAULT_ANOMALY_THRESHOLD)
    # Convert 2-D float heatmap to a colourised RGB image for overlay
    colored_anomaly = heatmap.apply_colormap(anomaly_heatmap, colormap=cv2.COLORMAP_HOT)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 – Run DenseNet anatomical analysis
    # ─────────────────────────────────────────────────────────────────────────
    progress_bar.progress(40, text=pipeline_steps[2])
    # DenseNet expects a 3-channel input – replicate the grayscale channel
    tensor_3ch = tensor_img.unsqueeze(0).repeat(1, 3, 1, 1)  # (1, 3, 224, 224)

    try:
        cnn_results  = cnn_analyzer.run_inference(cnn_model, tensor_3ch)
    except Exception as exc:
        st.warning(f"⚠️ DenseNet inference issue: {exc}. Proceeding with defaults.")
        cnn_results = {
            "top_labels": ["N/A"], "top_probs": [0.0],
            "top_idx": 0, "all_probs": {}, "feature_maps": None,
            "feature_vector": None,
        }

    top_prediction = cnn_results["top_labels"][0]
    top_prob       = cnn_results["top_probs"][0]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 – Run artifact detection
    # ─────────────────────────────────────────────────────────────────────────
    progress_bar.progress(60, text=pipeline_steps[3])
    artifact_results = artifact_detector.detect_artifacts(original_pil)
    has_artifacts    = artifact_results["artifact_detected"]
    artifact_count   = artifact_results["artifact_count"]
    bbox_list        = artifact_results["bounding_boxes"]
    annotated_image  = artifact_results["annotated_image"]  # pre-drawn RGB array

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 – Generate Grad-CAM heatmap
    # ─────────────────────────────────────────────────────────────────────────
    progress_bar.progress(80, text=pipeline_steps[4])
    target_layer = cnn_analyzer.get_target_layer(cnn_model)
    try:
        cam = heatmap.generate_gradcam(
            cnn_model, tensor_3ch,
            target_layer,
            class_idx=cnn_results["top_idx"],
        )
    except Exception:
        # Graceful fallback when grad-cam library not available
        cam = np.zeros((224, 224), dtype=np.float32)

    progress_bar.progress(100, text="✅ Pipeline complete!")

    # =========================================================================
    # STEP 8 – Display results
    # =========================================================================

    st.success("✅ Analysis complete!")

    # ── 8a. Summary metrics row ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Step 8 – Results")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric(
        "Top DenseNet Finding",
        top_prediction,
        f"{top_prob * 100:.1f}% confidence",
    )
    col_m2.metric(
        "Autoencoder Loss",
        f"{ae_loss:.5f}",
        "Above threshold ⚠️" if is_anomalous else "Within normal range ✅",
        delta_color="inverse" if is_anomalous else "normal",
    )
    col_m3.metric(
        "Artifacts Detected",
        artifact_count,
        "Detected ⚠️" if has_artifacts else "None ✅",
        delta_color="inverse" if has_artifacts else "normal",
    )
    col_m4.metric(
        "Anomaly Status",
        "Anomalous" if is_anomalous else "Normal",
        delta_color="inverse" if is_anomalous else "normal",
    )

    st.markdown("---")

    # ── 8b. Build composite overlay ───────────────────────────────────────────
    # Start from the original display and apply active layers sequentially
    final_overlay  = original_display.copy()
    layer_labels   = []

    if show_original:
        layer_labels.append("Original")
    else:
        final_overlay = np.zeros_like(final_overlay)    # black background

    if show_anomaly:
        final_overlay = heatmap.blend_images(final_overlay, colored_anomaly, alpha=0.35)
        layer_labels.append("Anomaly Map")

    if show_gradcam:
        final_overlay = heatmap.overlay_heatmap(cam, final_overlay, alpha=gradcam_opacity)
        layer_labels.append("Grad-CAM")

    if show_artifacts and has_artifacts:
        for (x, y, w, h) in bbox_list:
            cv2.rectangle(final_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                final_overlay, "Artifact",
                (x, max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1, cv2.LINE_AA,
            )
        layer_labels.append("Artifacts")

    view_title = " + ".join(layer_labels) if layer_labels else "Empty View"

    # ── 8c. Main image + patient profile ──────────────────────────────────────
    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ Interactive Layer View")
        st.image(final_overlay, caption=f"View: {view_title}", use_container_width=True)

    with col_info:
        st.subheader("🧑‍⚕️ Patient Profile")
        st.write(f"**ID:** {patient_id  if patient_id  else '—'}")
        st.write(f"**Age:** {int(patient_age) if patient_age else '—'}")
        st.write(f"**Sex:** {patient_sex if patient_sex != '—' else '—'}")
        st.write("**Symptoms / Notes:**")
        st.info(symptoms if symptoms else "None reported.")

        st.markdown("---")
        st.subheader("🔍 All DenseNet Predictions")
        for label, prob in zip(cnn_results["top_labels"], cnn_results["top_probs"]):
            bar_col, label_col = st.columns([3, 1])
            bar_col.progress(int(prob * 100))
            label_col.write(f"**{label}**")
            st.caption(f"{prob * 100:.2f}%")

    # ── 8d. Artifact detail panel ─────────────────────────────────────────────
    if has_artifacts:
        st.markdown("---")
        with st.expander("🔎 Artifact Detection Details", expanded=False):
            art_col1, art_col2 = st.columns(2)
            with art_col1:
                st.image(annotated_image,
                         caption="Artifact Annotation (from detector)",
                         use_container_width=True)
            with art_col2:
                st.write(f"**Total artifacts found:** {artifact_count}")
                st.write("**Bounding boxes (x, y, w, h):**")
                for i, box in enumerate(bbox_list, 1):
                    st.write(f"  {i}. x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")

    # ── 8e. Autoencoder reconstruction detail ─────────────────────────────────
    st.markdown("---")
    with st.expander("🔬 Autoencoder Reconstruction Details", expanded=False):
        ae_col1, ae_col2, ae_col3 = st.columns(3)
        with ae_col1:
            st.image(original_display,
                     caption="Original (pre-processed display)",
                     use_container_width=True)
        with ae_col2:
            recon_np = ae_reconstructed.squeeze().cpu().numpy()
            recon_np = np.clip(recon_np * 255, 0, 255).astype(np.uint8)
            st.image(recon_np, caption="AE Reconstruction", use_container_width=True)
        with ae_col3:
            st.image(colored_anomaly,
                     caption="Anomaly Map (HOT colormap)",
                     use_container_width=True)
        st.write(f"**Reconstruction loss (MSE):** `{ae_loss:.6f}`")
        threshold = autoencoder.DEFAULT_ANOMALY_THRESHOLD
        status    = "⚠️ ANOMALOUS" if is_anomalous else "✅ NORMAL"
        st.write(f"**Threshold:** `{threshold}` → **Status:** {status}")

else:
    # ── Placeholder shown before a file is uploaded ───────────────────────────
    st.info(
        "👆 Upload a chest X-ray image using the uploader above to start the "
        "AI analysis pipeline."
    )
    st.markdown(
        """
        **Pipeline overview:**
        | Step | Module | Description |
        |------|--------|-------------|
        | 1 | — | Collect patient input via sidebar |
        | 2 | — | Upload X-ray (PNG / JPG / DICOM) |
        | 3 | `preprocess` | Resize, normalise, convert to tensor |
        | 4 | `autoencoder` | Reconstruction error & anomaly map |
        | 5 | `cnn_analyzer` | DenseNet121 anatomical classification |
        | 6 | `artifact_detector` | Edge & contour-based artifact detection |
        | 7 | `heatmap` | Grad-CAM attention visualisation |
        | 8 | — | Display interactive results dashboard |
        """
    )
