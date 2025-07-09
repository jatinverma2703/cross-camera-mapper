import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Cross-Camera Player Matching Dashboard", layout="wide")
st.title("🎯 Cross-Camera Re-Identification Viewer")

# Load metrics
metrics_path = "outputs/evaluation_metrics.csv"
if os.path.exists(metrics_path):
    metrics = pd.read_csv(metrics_path)
    st.subheader("📊 Evaluation Metrics")
    st.dataframe(metrics.style.format("{:.4f}"))
else:
    st.warning("⚠️ Metrics file not found!")

# Match Viewer
st.subheader("🖼️ Matched Player Visuals")
match_dir = "outputs/match_visuals"
match_imgs = sorted([f for f in os.listdir(match_dir) if f.endswith('.jpg')])

selected = st.slider("Select Match Index", 0, len(match_imgs)-1, 0)
img_path = os.path.join(match_dir, match_imgs[selected])
st.image(Image.open(img_path), caption=f"Match {selected:03d}", use_container_width=True)

# TP / FP / FN browser
st.subheader("📂 Evaluation Visuals")

eval_type = st.radio("Choose Evaluation Type", ['tp', 'fp', 'fn'])
eval_dir = os.path.join("outputs/evaluation_visuals", eval_type)
eval_imgs = sorted([f for f in os.listdir(eval_dir) if f.endswith('.jpg')])

if eval_imgs:
    eval_idx = st.slider(f"View {eval_type.upper()} image index", 0, len(eval_imgs)-1, 0)
    eval_path = os.path.join(eval_dir, eval_imgs[eval_idx])
    st.image(Image.open(eval_path), caption=f"{eval_type.upper()} Match {eval_idx}", use_container_width=True)
else:
    st.warning(f"No images found in {eval_dir}")