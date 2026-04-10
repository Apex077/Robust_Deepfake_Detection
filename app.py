import streamlit as st
import torch
import yaml
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import mtcnn

# Setup paths so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid_net import HybridSwinNet
from src.datasets.augmentations import build_val_transform

st.set_page_config(
    page_title="Deepfake Detection Showcase",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .result-fake {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 24px;
    }
    .result-real {
        color: #00c04b;
        font-weight: bold;
        font-size: 24px;
    }
    .gt-text {
        font-size: 20px;
        color: #666;
    }
    .info-box {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)


def _load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_model_and_transform(
    checkpoint_path: str = "checkpoints/model_best.pth",
    config_path: str = "configs/default.yaml",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = _load_config(config_path)
    model_cfg = config["model"]

    model = HybridSwinNet(
        swinv2_variant=model_cfg["swinv2_variant"],
        pretrained=False,
        freq_embed_dim=model_cfg.get("freq_embed_dim", 512),
        freq_branch_dim=model_cfg.get("freq_branch_dim", 256),
        fmsi_mask_ratio=0.0,          # Always 0 at inference (no DCT masking)
        fusion_d_model=model_cfg.get("fusion_d_model", 512),
        fusion_heads=model_cfg.get("fusion_heads", 8),
        fusion_dropout=model_cfg.get("fusion_dropout", 0.1),
    ).to(device)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return None, None, device, {}

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            st.warning(f"Missing keys when loading checkpoint: {missing[:5]}")
        if unexpected:
            st.warning(f"Unexpected keys when loading checkpoint: {unexpected[:5]}")
        model.eval()

        meta = {
            "epoch": ckpt.get("epoch", "N/A"),
            "best_auc": ckpt.get("best_auc", None),
            "checkpoint": checkpoint_path,
            "device": str(device),
        }
    except Exception as e:
        st.error(f"Error loading checkpoint at {checkpoint_path}: {e}")
        return None, None, device, {}

    transform = build_val_transform()
    return model, transform, device, meta


@st.cache_resource
def get_face_detector():
    """Cache the MTCNN initialization to avoid massive overhead per inference."""
    return mtcnn.MTCNN()


def preprocess_face(image_np, detector):
    """
    Detect the largest face and crop it, preserving high-frequency features
    prior to resizing for the neural network.
    Returns:
        cropped_face: (H, W, 3) cropped face if found, else original image
        face_box: tuple of (x, y, w, h) if found, else None
    """
    faces = detector.detect_faces(image_np)
    if faces:
        # Find largest face to avoid background faces/false positives
        best_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = best_face['box']
        
        # Ensure bounds don't go negative
        x, y = max(0, x), max(0, y)
        face_crop = image_np[y:y+h, x:x+w]
        
        # We don't need cv2.resize(256,256) here because build_val_transform() 
        # executes A.Resize(256, 256) which handles it properly inside Albumentations.
        return face_crop, (x, y, w, h)
    return image_np, None



def _load_thresholds() -> dict:
    """
    Load threshold info from the overfitting analysis JSON.

    Returns a dict with keys:
        val_optimal  – threshold maximising F1 on the internal validation split
        val_auc      – AUC on internal validation split
        val_recall   – recall on validation at val_optimal threshold
    or empty dict if file not found.
    """
    analysis_path = Path("results/overfitting/overfitting_analysis.json")
    if not analysis_path.exists():
        return {}
    try:
        with open(analysis_path) as f:
            data = json.load(f)
        val = data.get("validation", {})
        return {
            "val_optimal": val.get("optimal_threshold"),
            "val_auc": val.get("overall_auc"),
            "val_recall": val.get("recall"),
            "val_precision": val.get("precision"),
            "val_accuracy": val.get("accuracy"),
        }
    except Exception:
        return {}


def main():
    st.title("🛡️ Robust Deepfake Detection")
    st.markdown("### HybridSwinNet — Dual-stream spatial + frequency analysis")

    # Load model
    model, transform, device, meta = get_model_and_transform()

    if model is None:
        st.warning("Please ensure a valid checkpoint is placed at `checkpoints/model_best.pth`.")
        return

    # ------------------------------------------------------------------ #
    # Sidebar — controls + model info                                      #
    # ------------------------------------------------------------------ #
    st.sidebar.header("Controls")

    # Threshold logic
    # Default is 0.5 (balanced).  The val-optimal threshold (0.83) is shown
    # as an option but NOT used by default — it has 68 % recall, meaning it
    # misses ~32 % of fakes when tested on out-of-distribution images.
    thresh_info = _load_thresholds()
    val_optimal = thresh_info.get("val_optimal")

    threshold_options = {
        "0.50 — Balanced (default)": 0.50,
        "0.40 — Higher sensitivity (fewer missed fakes)": 0.40,
        "0.30 — Maximum sensitivity": 0.30,
    }
    if val_optimal is not None:
        label = f"{val_optimal:.2f} — Val-optimal (high precision, recall ≈ {thresh_info.get('val_recall', 0):.0%})"
        threshold_options[label] = float(val_optimal)

    selected_thresh_label = st.sidebar.selectbox(
        "Decision threshold (Fake ≥ limit):",
        list(threshold_options.keys()),
        index=0,
        help=(
            "Lower = more sensitive (catches more fakes, more false alarms). "
            "The val-optimal threshold maximises F1 on the internal eval split "
            "but has lower recall on out-of-distribution deepfakes."
        ),
    )
    threshold = threshold_options[selected_thresh_label]

    # Fine-tune slider
    threshold = st.sidebar.slider(
        "Fine-tune threshold:",
        min_value=0.0,
        max_value=1.0,
        value=threshold,
        step=0.01,
    )

    # Model info panel
    with st.sidebar.expander("ℹ️ Model Info", expanded=False):
        st.write(f"**Checkpoint epoch:** {meta.get('epoch', 'N/A')}")
        if meta.get("best_auc"):
            st.write(f"**Train best AUC:** {meta['best_auc']:.4f}")
        if thresh_info.get("val_auc"):
            st.write(f"**Val AUC (internal):** {thresh_info['val_auc']:.4f}")
        if thresh_info.get("val_recall"):
            st.write(f"**Val recall @ {val_optimal:.2f}:** {thresh_info['val_recall']:.1%}")
        st.write(f"**Device:** {meta.get('device', 'N/A')}")
        st.write(f"**File:** {meta.get('checkpoint', 'N/A')}")

    # ------------------------------------------------------------------ #
    # Dataset image selection                                              #
    # ------------------------------------------------------------------ #
    data_dir = Path("data/trainval_data_final/training_data_final")

    if data_dir.exists():
        all_images = sorted(
            [f for f in data_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
    else:
        all_images = []

    ground_truth = "Unknown"

    uploaded_file = st.sidebar.file_uploader("Upload an Image to analyze:", type=["png", "jpg", "jpeg"])

    if all_images:
        if "selector" not in st.session_state:
            st.session_state.selector = all_images[0].name

        def pick_random():
            import random
            st.session_state.selector = random.choice(all_images).name

        st.sidebar.button("🎲 Pick Random Dataset Image", on_click=pick_random, use_container_width=True, disabled=(uploaded_file is not None))
        st.sidebar.selectbox(
            "Select an image from dataset:",
            [img.name for img in all_images],
            key="selector",
            disabled=(uploaded_file is not None)
        )
    else:
        if "selector" in st.session_state:
            del st.session_state["selector"]

    # ------------------------------------------------------------------ #
    # Load image                                                           #
    # ------------------------------------------------------------------ #
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            ground_truth = "Unknown"
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            return
    elif all_images and "selector" in st.session_state:
        selected_path = data_dir / st.session_state.selector
        name_lower = st.session_state.selector.lower()
        if "_real" in name_lower:
            ground_truth = "REAL"
        elif "_fake" in name_lower:
            ground_truth = "FAKE"
        try:
            image = Image.open(selected_path).convert("RGB")
        except Exception as e:
            st.error(f"Error loading dataset image: {e}")
            return
    else:
        st.info("No dataset images found. Please upload an image using the sidebar.")
        return

    # ------------------------------------------------------------------ #
    # Display + inference                                                  #
    # ------------------------------------------------------------------ #
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Input Image")
        st.image(image, use_container_width=True)
        if ground_truth != "Unknown":
            gt_class = "result-fake" if ground_truth == "FAKE" else "result-real"
            st.markdown(
                f"Ground truth: <span class='{gt_class}'>{ground_truth}</span>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### Model Analysis")

        if st.button("▶ Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Analysing image…"):
                # Pre-process
                img_np = np.array(image, dtype=np.uint8)
                
                # Face detection and crop
                detector = get_face_detector()
                face_np, box = preprocess_face(img_np, detector)
                
                if box:
                    st.success(f"Detected face for analysis at {box}")
                    st.image(face_np, caption="Cropped Face Input", use_container_width=False, width=150)
                else:
                    st.warning("No face detected! Running full image analysis.")
                
                # Transform includes Resize(256, 256) and ImageNet Normalisation
                tensor = transform(image=face_np)["image"].unsqueeze(0).to(device)

                # Forward pass — model.forward() returns raw logits
                with torch.no_grad():
                    logit = model(tensor).item()
                    prob = float(torch.sigmoid(torch.tensor(logit)))

            pred_label = "FAKE" if prob >= threshold else "REAL"

            st.markdown("---")
            st.markdown("### Results")

            res_class = "result-fake" if pred_label == "FAKE" else "result-real"
            st.markdown(
                f"Prediction: <span class='{res_class}'>{pred_label}</span>",
                unsafe_allow_html=True,
            )

            st.progress(prob, text=f"Fake probability: {prob:.2%}")

            if prob < threshold and prob >= 0.40:
                st.warning(
                    f"⚠️ Fake probability ({prob:.2%}) is below the threshold "
                    f"({threshold:.2f}) but above 40 %. Consider lowering the "
                    "threshold to catch more fakes at the cost of more false alarms."
                )

            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            c1.metric("Fake Probability", f"{prob:.4f}")
            c2.metric("Raw Logit", f"{logit:.4f}")
            c3.metric("Threshold", f"{threshold:.2f}")

            if ground_truth != "Unknown":
                st.markdown("---")
                correct = pred_label == ground_truth
                if correct:
                    st.success(f"✅ Correct — ground truth is {ground_truth}")
                else:
                    st.error(f"❌ Wrong — ground truth is {ground_truth}, predicted {pred_label}")


if __name__ == "__main__":
    main()
