import streamlit as st
import torch
import yaml
import sys
import numpy as np
from PIL import Image
from pathlib import Path

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
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_model_and_transform(checkpoint_path="checkpoints/model_best.pth", config_path="configs/default.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    model_cfg = config["model"]
    model = HybridSwinNet(
        swinv2_variant=model_cfg["swinv2_variant"],
        pretrained=False,
        freq_embed_dim=model_cfg.get("freq_embed_dim", 512),
        freq_branch_dim=model_cfg.get("freq_branch_dim", 256),
        fmsi_mask_ratio=0.0,
        fusion_d_model=model_cfg.get("fusion_d_model", 512),
        fusion_heads=model_cfg.get("fusion_heads", 8),
    ).to(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        st.error(f"Error loading checkpoint at {checkpoint_path}: {e}")
        return None, None, device

    transform = build_val_transform()
    return model, transform, device

def main():
    st.title("🛡️ Robust Deepfake Detection")
    st.markdown("### Model prediction and ground truth presentation showcase.")

    # Main content setup
    model, transform, device = get_model_and_transform()
    
    if model is None:
        st.warning("Please ensure a valid checkpoint is placed at `checkpoints/model_best.pth`.")
        return

    data_dir = Path("data/trainval_data_final/training_data_final")
    
    if not data_dir.exists():
        st.error(f"Dataset directory not found: {data_dir}. Cannot load images for the showcase.")
        return

    # Get images
    all_images = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    if len(all_images) == 0:
        st.warning(f"No images found in {data_dir}")
        return
        
    st.sidebar.header("Controls")
    
    # Image selection
    selected_filename = st.sidebar.selectbox("Select an Image:", [img.name for img in all_images])
    selected_path = data_dir / selected_filename
    
    # Determine Ground Truth from filename
    ground_truth = "Unknown"
    is_real = "_real" in selected_filename.lower()
    is_fake = "_fake" in selected_filename.lower()
    
    if is_real:
        ground_truth = "REAL"
    elif is_fake:
        ground_truth = "FAKE"
        
    
    # Load and display selected image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Input Image")
        try:
            image = Image.open(selected_path).convert("RGB")
            st.image(image, use_container_width=True)
            st.markdown(f"<div class='gt-text'>Ground Truth: <strong>{ground_truth}</strong></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
            
    with col2:
        st.markdown("#### Model Analysis")
        
        threshold = st.sidebar.slider("Decision Threshold (Fake >= limit)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        if st.button("Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                img_np = np.array(image, dtype=np.uint8)
                tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).item()

                pred_label = "FAKE" if prob >= threshold else "REAL"
                
                st.markdown("---")
                st.markdown("### Results")
                
                # Layout based on results
                res_class = "result-fake" if pred_label == "FAKE" else "result-real"
                st.markdown(f"Prediction: <span class='{res_class}'>{pred_label}</span>", unsafe_allow_html=True)
                
                st.progress(prob, text=f"Confidence of being Fake: {prob:.2%}")
                
                # Ground truth comparison
                if ground_truth != "Unknown":
                    if ground_truth == pred_label:
                        st.success(f"✅ Correct prediction! Both are {ground_truth}.")
                    else:
                        st.error(f"❌ Incorrect prediction. Expected {ground_truth}, got {pred_label}.")
                        
                st.markdown("---")
                
                col_metric1, col_metric2 = st.columns(2)
                col_metric1.metric("Fake Probability", f"{prob:.4f}")
                col_metric2.metric("Threshold Used", f"{threshold:.2f}")

if __name__ == "__main__":
    main()
