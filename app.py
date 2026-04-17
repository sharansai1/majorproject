"""
Pancreatic Tumor Detection & Localization — Streamlit App
Hybrid GAT U-Net for CT Scan Tumor Segmentation

Deploy to Streamlit Cloud:
1. Push this folder to GitHub
2. Go to https://share.streamlit.io
3. New App → Select Repo → Select app.py → Deploy
"""

import base64
import io
from pathlib import Path

import albumentations as A
import groq
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import HybridGATUNet

# Configuration
MODEL_PATH = Path(__file__).resolve().parent / "best_gat_unet.pth"
IMAGE_SIZE = 256


# ── Helper Functions ─────────────────────────────────────────────────────────

def get_device():
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model():
    """Load the trained model with caching for performance."""
    device = get_device()
    model = HybridGATUNet().to(device)
    
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure 'best_gat_unet.pth' is in the app folder.")
        st.stop()
    
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    )
    model.eval()
    return model, device


def preprocess(image_gray):
    """Preprocess a grayscale CT image for model inference."""
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return transform(image=image_rgb)["image"].unsqueeze(0).float()


def run_inference(model, tensor, device):
    """Run model inference and return probability heatmap."""
    with torch.no_grad():
        logits = model(tensor.to(device))
        return torch.sigmoid(logits).squeeze().cpu().numpy()


def get_contours(mask):
    """Extract contours from binary mask."""
    m = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours_on_ax(ax, contours, color, linewidth=2, label=None):
    """Draw contours on matplotlib axes."""
    from matplotlib.patches import Polygon
    for i, cnt in enumerate(contours):
        if len(cnt) < 3:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2:
            continue
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=color,
                             linewidth=linewidth, label=label if i == 0 else None))


def build_result_figure(ct, heatmap, pred_binary):
    """Build the visualization figure with input, heatmap, and segmentation."""
    pred_contours = get_contours(pred_binary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Pancreatic Tumor Detection & Localization", fontsize=14, fontweight="bold")

    # Input CT Scan
    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input CT Scan", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Prediction Heatmap
    axes[1].imshow(ct, cmap="gray", vmin=0, vmax=1)
    im = axes[1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Tumor Probability Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("Probability", fontsize=9)

    # Segmentation Result
    axes[2].imshow(ct, cmap="gray", vmin=0, vmax=1)
    pred_ov = np.zeros((*ct.shape, 4))
    pred_ov[pred_binary > 0] = [0.0, 1.0, 0.5, 0.4]  # Green overlay
    axes[2].imshow(pred_ov)
    draw_contours_on_ax(axes[2], pred_contours, "lime", 2.5)
    axes[2].set_title("Tumor Localization", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Add stats text
    tumor_pct = pred_binary.sum() / pred_binary.size * 100
    axes[2].text(
        0.02, 0.02,
        f"Tumor pixels: {int(pred_binary.sum())}\n"
        f"Coverage: {tumor_pct:.2f}%\n"
        f"Max confidence: {heatmap.max():.2%}",
        transform=axes[2].transAxes, fontsize=9, color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75),
    )

    plt.tight_layout()
    return fig


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# ── AI Diagnosis Functions ───────────────────────────────────────────────────

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def get_groq_client():
    """Get Groq client using Streamlit secrets."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            return groq.Groq(api_key=api_key)
    except Exception:
        pass
    return None


def pil_to_base64(pil_image):
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_diagnosis_prompt(tumor_detected, tumor_pixels, coverage_pct, max_confidence, threshold, bbox=None):
    """Build prompt for AI diagnosis."""
    parts = [
        "You are an expert radiologist AI assistant specializing in pancreatic imaging. ",
        "A CT scan slice of the abdomen has been analyzed by a Hybrid GAT U-Net deep-learning "
        "model trained on the Medical Segmentation Decathlon Task07 Pancreas dataset.\n\n",
        "## Model Analysis Results\n",
        f"- **Tumor Detected**: {'Yes' if tumor_detected else 'No'}\n",
        f"- **Tumor Pixels**: {tumor_pixels:,}\n",
        f"- **Tumor Coverage**: {coverage_pct:.2f}% of the slice\n",
        f"- **Maximum Confidence**: {max_confidence:.1%}\n",
        f"- **Detection Threshold**: {threshold}\n",
    ]

    if bbox:
        parts.append(
            f"- **Bounding Box**: X [{bbox['col_min']}-{bbox['col_max']}], "
            f"Y [{bbox['row_min']}-{bbox['row_max']}], "
            f"Size {bbox['width']}x{bbox['height']} px\n"
        )

    parts.append(
        "\n## Instructions\n"
        "Based on the CT scan image and the model's analysis above, provide:\n"
        "1. **Clinical Assessment**: Interpret the findings — is the tumor likely "
        "malignant or benign based on size, location, and morphology visible in the scan?\n"
        "2. **Possible Diagnoses**: List the most probable diagnoses (e.g., pancreatic "
        "ductal adenocarcinoma, neuroendocrine tumor, cystic neoplasm, etc.) with brief reasoning.\n"
        "3. **Severity Estimation**: Low / Moderate / High concern, and why.\n"
        "4. **Recommended Next Steps**: What additional imaging, labs, or procedures "
        "should be considered?\n"
        "5. **Model Confidence Commentary**: Comment on how confident the model appears "
        "and any caveats about false positives/negatives.\n\n"
        "**Important**: Clearly state that this is an AI-assisted preliminary analysis "
        "and NOT a substitute for professional medical diagnosis. "
        "Recommend consultation with a qualified radiologist/oncologist.\n"
        "Format your response with clear headings and bullet points."
    )

    return "".join(parts)


def get_ai_diagnosis(client, ct_image_pil, prompt):
    """Get AI diagnosis from Groq API."""
    img_b64 = pil_to_base64(ct_image_pil)
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pancreas Tumor Detection",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    .main-header { 
        font-size: 2rem; 
        font-weight: 700; 
        color: #1a73e8; 
        margin-bottom: 0.2rem; 
    }
    .sub-header { 
        font-size: 1.1rem; 
        color: #555; 
        margin-bottom: 1.5rem; 
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px; 
        padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8; 
        margin-bottom: 0.6rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #666; }
    .metric-card p { margin: 0; font-size: 1.5rem; font-weight: 700; color: #1a73e8; }
    .detection-positive { border-left-color: #e53935; }
    .detection-positive p { color: #e53935; }
    .detection-negative { border-left-color: #43a047; }
    .detection-negative p { color: #43a047; }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    .ai-diagnosis-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border: 1px solid #a8c7fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .ai-diagnosis-box h4 { color: #1a56db; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Pancreatic Tumor Detection & Localization</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid GAT U-Net &nbsp;|&nbsp; AI-powered CT scan analysis for tumor segmentation</div>',
    unsafe_allow_html=True,
)

# Load model
model, device = load_model()
st.sidebar.success(f"Model loaded on **{str(device).upper()}**")

# Initialize Groq client (if API key is configured in secrets)
groq_client = get_groq_client()


# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown(
    "1. Upload a CT scan slice (PNG/JPG) below\n"
    "2. Adjust detection threshold if needed\n"
    "3. View tumor detection results\n"
    "4. Click 'Get AI Diagnosis' for expert analysis\n"
    "5. Download the report image"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(
    "- **Architecture**: Hybrid GAT U-Net\\n"
    "- **Input**: 256 x 256 grayscale CT\\n"
    "- **Output**: Tumor probability map\\n"
    "- **Dataset**: MSD Task07 Pancreas"
)
st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Detection Threshold", 
    0.1, 0.9, 0.5, 0.05,
    help="Pixels with probability above this are classified as tumor."
)


# ── Main Content ─────────────────────────────────────────────────────────────

# Info box
st.markdown("""
<div class="info-box">
<strong>Instructions:</strong> Upload a preprocessed CT scan slice image. 
The model will analyze it and detect any pancreatic tumors present. 
For best results, use grayscale PNG images from CT scans.
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload CT Scan Slice",
    type=["png", "jpg", "jpeg"],
    help="Upload a CT scan image (grayscale preferred)"
)

if uploaded_file is not None:
    # Read and process image
    ct_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    ct_raw = cv2.imdecode(ct_bytes, cv2.IMREAD_GRAYSCALE)

    if ct_raw is None:
        st.error("Could not read the uploaded image. Please try a different file.")
    else:
        with st.spinner("Running AI inference... This may take a few seconds."):
            # Preprocess and run inference
            input_tensor = preprocess(ct_raw)
            heatmap = run_inference(model, input_tensor, device)
            pred_binary = (heatmap > threshold).astype(np.float32)

        # Resize original for display
        ct_resized = cv2.resize(ct_raw, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        tumor_detected = pred_binary.sum() > 0
        tumor_pct = pred_binary.sum() / pred_binary.size * 100

        st.markdown("---")

        # Detection result banner
        if tumor_detected:
            st.markdown(
                '<div class="metric-card detection-positive">'
                '<h3>DETECTION RESULT</h3>'
                '<p>TUMOR DETECTED</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card detection-negative">'
                '<h3>DETECTION RESULT</h3>'
                '<p>NO TUMOR DETECTED</p></div>',
                unsafe_allow_html=True,
            )

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>Tumor Pixels</h3>'
                f'<p>{int(pred_binary.sum()):,}</p></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><h3>Coverage</h3>'
                f'<p>{tumor_pct:.2f}%</p></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><h3>Max Confidence</h3>'
                f'<p>{heatmap.max():.1%}</p></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><h3>Threshold</h3>'
                f'<p>{threshold}</p></div>',
                unsafe_allow_html=True,
            )

        # Visualization
        st.markdown("#### Analysis Results")
        fig = build_result_figure(ct_resized, heatmap, pred_binary)
        report_img = fig_to_image(fig)
        plt.close(fig)
        st.image(report_img, use_container_width=True)

        # Localization info
        if tumor_detected:
            coords = np.where(pred_binary > 0)
            row_min, row_max = int(coords[0].min()), int(coords[0].max())
            col_min, col_max = int(coords[1].min()), int(coords[1].max())

            st.markdown("#### Tumor Localization (Bounding Box)")
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.metric("X range (px)", f"{col_min} - {col_max}")
            with l2:
                st.metric("Y range (px)", f"{row_min} - {row_max}")
            with l3:
                st.metric("Width (px)", f"{col_max - col_min + 1}")
            with l4:
                st.metric("Height (px)", f"{row_max - row_min + 1}")

        # AI Diagnosis Section
        st.markdown("---")
        st.markdown("#### AI-Powered Diagnosis (Groq)")

        if not groq_client:
            st.info(
                "Set the `GROQ_API_KEY` in Streamlit Secrets to enable AI diagnosis. "
                "Get a free key at [console.groq.com/keys](https://console.groq.com/keys).\n\n"
                "To add secrets:\n"
                "1. Go to your app dashboard on Streamlit Cloud\n"
                "2. Click Settings → Secrets\n"
                "3. Add: `GROQ_API_KEY = 'your-key-here'`"
            )
        else:
            bbox_info = None
            if tumor_detected:
                coords = np.where(pred_binary > 0)
                bbox_info = {
                    "row_min": int(coords[0].min()),
                    "row_max": int(coords[0].max()),
                    "col_min": int(coords[1].min()),
                    "col_max": int(coords[1].max()),
                    "width": int(coords[1].max() - coords[1].min() + 1),
                    "height": int(coords[0].max() - coords[0].min() + 1),
                }

            if st.button("Get AI Diagnosis", type="primary", key="ai_diag"):
                with st.spinner("Consulting Llama AI via Groq..."):
                    prompt = build_diagnosis_prompt(
                        tumor_detected=tumor_detected,
                        tumor_pixels=int(pred_binary.sum()),
                        coverage_pct=tumor_pct,
                        max_confidence=float(heatmap.max()),
                        threshold=threshold,
                        bbox=bbox_info,
                    )
                    ct_gray = (ct_resized * 255).astype(np.uint8)
                    ct_rgb = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2RGB)
                    ct_pil = Image.fromarray(ct_rgb)
                    try:
                        diagnosis = get_ai_diagnosis(groq_client, ct_pil, prompt)
                        st.session_state["last_diagnosis"] = diagnosis
                    except Exception as exc:
                        st.error(f"Groq API error: {exc}")
                        st.session_state.pop("last_diagnosis", None)

            if "last_diagnosis" in st.session_state:
                st.markdown(
                    '<div class="ai-diagnosis-box">',
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["last_diagnosis"])
                st.markdown("</div>", unsafe_allow_html=True)

        # Download button
        st.markdown("---")
        buf = io.BytesIO()
        report_img.save(buf, format="PNG")
        st.download_button(
            label="Download Report Image",
            data=buf.getvalue(),
            file_name=f"tumor_analysis_{uploaded_file.name}",
            mime="image/png",
        )


# ── Footer ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Hybrid GAT U-Net for Pancreatic Tumor Detection<br>"
    "Medical Segmentation Decathlon Task07 Pancreas</small>",
    unsafe_allow_html=True,
)
