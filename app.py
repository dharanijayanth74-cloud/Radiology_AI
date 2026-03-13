import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch

# ── Module imports ────────────────────────────────────────────────────────────
import preprocess      
import cnn_analyzer    
import heatmap             

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RadAI - Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 RadAI - Chest X-Ray Analysis")
st.markdown(
    "Upload a chest X-ray to perform DenseNet medical classification, and visualize Grad-CAM activation maps."
)

# ═════════════════════════════════════════════════════════════════════════════
# Collect patient input
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.header("🏥 Patient Data")
patient_id = st.sidebar.text_input("Patient ID", placeholder="e.g. 12345")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
symptoms = st.sidebar.text_area("Clinical Symptoms", placeholder="Patient notes...")

st.sidebar.markdown("---")
st.sidebar.header("👁️ Visualization Toggles")
show_original = st.sidebar.checkbox("Show Original Image", value=True)
show_gradcam = st.sidebar.checkbox("Overlay Grad-CAM Heatmap", value=True)

st.sidebar.markdown("---")
gradcam_opacity = st.sidebar.slider("Grad-CAM Opacity", 0.1, 0.9, 0.5, 0.1)

# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading DenseNet121 Medical model...")
def load_cnn():
    return cnn_analyzer.build_densenet()

try:
    cnn_model = load_cnn()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# ═════════════════════════════════════════════════════════════════════════════
# Upload X-ray
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
uploaded_file = st.file_uploader(
    "Accepted formats: PNG, JPG, JPEG, DICOM (.dcm)",
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file is not None and models_loaded:
    with st.spinner("Processing Pipeline..."):
        # 1. Preprocess image
        try:
            tensor_img, original_pil = preprocess.preprocess_image(uploaded_file)
        except Exception as exc:
            st.error(f"❌ Preprocessing failed: {exc}")
            st.stop()

        original_np = np.array(original_pil)
        if len(original_np.shape) == 2:
            original_display = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        else:
            original_display = cv2.cvtColor(original_np, cv2.COLOR_RGBA2RGB) if original_np.shape[2] == 4 else original_np.copy()
            
        final_overlay = original_display.copy()

        # Ensure tensor is 3 channels for torch vision DenseNet mapping 
        if tensor_img.shape[1] == 1:
            tensor_3ch = tensor_img.repeat(1, 3, 1, 1)
        else:
            tensor_3ch = tensor_img

        # 3. DenseNet inference
        cnn_results = cnn_analyzer.run_inference(cnn_model, tensor_3ch)
        top_prediction = cnn_results["top_labels"][0] if cnn_results["top_labels"] else "No Finding"
        top_prob = cnn_results["top_probs"][0] if cnn_results["top_probs"] else 0.0

        # 4. Grad-CAM
        target_layer = cnn_analyzer.get_target_layer(cnn_model)
        try:
            cam = heatmap.generate_gradcam(
                cnn_model, tensor_3ch,
                target_layer,
                class_idx=cnn_results["top_idx"],
            )
        except Exception as e:
            cam = np.zeros((224, 224), dtype=np.float32)

    st.success("✅ Analysis complete!")

    # ═════════════════════════════════════════════════════════════════════════════
    # Display results
    # ═════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    col1.metric("Leading Finding", top_prediction, f"{top_prob * 100:.1f}% confidence")

    # Composite overlay
    layer_labels = []
    
    if show_original:
        layer_labels.append("Original")
    else:
        final_overlay = np.zeros_like(final_overlay)

    if show_gradcam:
        final_overlay = heatmap.overlay_heatmap(cam, final_overlay, alpha=gradcam_opacity)
        layer_labels.append("Grad-CAM")

    view_title = " + ".join(layer_labels) if layer_labels else "Empty View"

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ Interactive Layer View")
        st.image(final_overlay, caption=f"View: {view_title}", use_container_width=True)

    with col_info:
        st.subheader("🧑‍⚕️ Patient Profile")
        st.write(f"**ID:** {patient_id if patient_id else '—'}")
        st.write(f"**Age:** {int(patient_age) if patient_age else '—'}")
        st.write("**Symptoms / Notes:**")
        st.info(symptoms if symptoms else "None reported.")

        st.markdown("---")
        st.subheader("🔍 All Top-5 Predictions")
        
        if top_prediction == "No Finding" and top_prob == 0.0:
            st.success("No significant pathologies detected.")
        else:
            for label, prob in zip(cnn_results["top_labels"], cnn_results["top_probs"]):
                bar_col, label_col = st.columns([3, 1])
                bar_col.progress(int(prob * 100))
                label_col.write(f"**{label}**")
                st.caption(f"{prob * 100:.2f}%")
else:
    if not models_loaded:
        st.stop()
    st.info("👆 Upload a chest X-ray image to start the AI analysis pipeline.")
    st.markdown(
        """
        **Pipeline overview:**
        | Step | Module | Description |
        |------|--------|-------------|
        | 1 | `preprocess` | Resize, normalise, convert to tensor |
        | 2 | `cnn_analyzer.densenet` | Medical DenseNet121 anomaly classification |
        | 3 | `heatmap` | Grad-CAM attention visualisation |
        | 4 | UI | Display interactive results dashboard |
        """
    )
