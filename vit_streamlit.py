"""Streamlit app for ViT tiny binary classifier (NonDemented vs VeryMildDemented).

Assumes the best checkpoint was produced by `vit.py` and saved at
`models/tuned_v2/vit_ti_best_kfold_model.pth` relative to this file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from timm import create_model
from torchvision import transforms
from matplotlib import cm
from pytorch_grad_cam import GradCAM

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "tuned_v2" / "vit_ti_best_kfold_model.pth"
MODEL_NAME = "vit_tiny_patch16_224"
CLASS_NAMES = ["NonDemented", "VeryMildDemented"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_vit_reshape(grid_size: Tuple[int, int]):
    gh, gw = grid_size

    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [B, tokens, C]
        if tensor.ndim != 3:
            return tensor
        tensor = tensor[:, 1:, :]  # drop CLS
        B, N, C = tensor.shape
        if N != gh * gw:
            return tensor
        return tensor.reshape(B, gh, gw, C).permute(0, 3, 1, 2)

    return reshape_transform


def load_image(uploaded_file) -> Image.Image:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not read image: {exc}")
        raise


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = build_transform()
    return transform(img).unsqueeze(0)


def compute_vit_gradcam(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray | None:
    grid_size = getattr(model.patch_embed, "grid_size", None)
    if grid_size is None:
        return None

    reshape_transform = make_vit_reshape(grid_size)
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )
    grayscale_cam = cam(input_tensor=image_tensor, eigen_smooth=False, aug_smooth=False)
    if grayscale_cam is None or len(grayscale_cam) == 0:
        return None
    heatmap = np.clip(grayscale_cam[0], 0.0, 1.0)
    return heatmap


def colorize_and_overlay(image: Image.Image, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # heatmap: [H, W] in [0,1]
    heatmap = np.clip(heatmap, 0.0, 1.0)
    img_np = np.array(image.convert("RGB"))
    heat_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    heat_resized_np = np.array(heat_resized) / 255.0
    heat_color = (cm.jet(heat_resized_np)[..., :3] * 255).astype(np.uint8)
    overlay = (
        0.5 * img_np.astype(np.float32) + 0.5 * heat_color.astype(np.float32)
    ).astype(np.uint8)
    return heat_resized_np, overlay


@st.cache_resource(show_spinner="Loading ViT model...")
def load_model() -> nn.Module | None:
    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found at: {MODEL_PATH}")
        return None
    try:
        model = create_model(MODEL_NAME, pretrained=False, num_classes=1)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load model: {exc}")
        return None


def predict_with_cam(model: nn.Module, image_tensor: torch.Tensor) -> Tuple[float, str, np.ndarray | None]:
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)[0].item()

    heatmap = compute_vit_gradcam(model, image_tensor)
    label_idx = int(prob >= 0.5)
    return prob, CLASS_NAMES[label_idx], heatmap


# --- UI ---
st.set_page_config(page_title="ViT Dementia Classifier", layout="centered")
st.title("ViT Dementia Classifier")
st.caption(
    f"Model: {MODEL_NAME} | Device: {DEVICE} | Checkpoint: {MODEL_PATH.relative_to(SCRIPT_DIR)}"
)

if st.button("BALON!!!!"):
    st.balloons()

model = load_model()
if model is None:
    st.stop()

uploaded = st.file_uploader("Upload a brain scan image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded is not None:
    image = load_image(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Running inference..."):
        tensor = preprocess(image)
        prob, pred_label, heatmap = predict_with_cam(model, tensor)

    st.subheader("Prediction")
    st.metric(label="Predicted class", value=pred_label)
    st.metric(label="Probability (VeryMildDemented)", value=f"{prob*100:.2f}%")

    if heatmap is not None:
        heat_resized, overlay = colorize_and_overlay(image, heatmap)
        col1, col2 = st.columns(2)
        with col1:
            st.image(heat_resized, clamp=True, caption="Attention heatmap", use_column_width=True)
        with col2:
            st.image(overlay, caption="Overlay (image + heatmap)", use_column_width=True)
    else:
        st.info("No attention heatmap could be generated for this image.")
else:
    st.info("Upload an image to get a prediction.")

