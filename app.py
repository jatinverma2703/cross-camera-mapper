import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
import io
import zipfile
from sklearn.metrics import average_precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

#Config

MATCH_CSV = 'outputs/player_matches.csv'
GT_CSV = 'outputs/ground_truth_matches.csv'
BAD_LOG_CSV = 'outputs/bad_predictions_log.csv'
VISUALS_DIR = 'outputs/match_visuals'
DETECTIONS_DIR = 'outputs/detections'
EVAL_METRICS = 'outputs/evaluation_metrics.csv'
EVAL_VISUALS = 'outputs/evaluation_visuals'
VIDEO_PATHS = {
    'broadcast': 'data/broadcast.mp4',
    'tacticam': 'data/tacticam.mp4'
}

#Helper Functions 

def auto_generate_ground_truth():
    if os.path.exists(GT_CSV) and os.path.getsize(GT_CSV) > 0:
        return
    b_dir = os.path.join(DETECTIONS_DIR, 'broadcast')
    t_dir = os.path.join(DETECTIONS_DIR, 'tacticam')
    b_imgs = sorted([f for f in os.listdir(b_dir) if f.endswith('.jpg')])
    t_imgs = sorted([f for f in os.listdir(t_dir) if f.endswith('.jpg')])
    n = min(len(b_imgs), len(t_imgs))
    df = pd.DataFrame(zip(b_imgs[:n], t_imgs[:n]), columns=['Broadcast Image', 'Tacticam Match'])
    df.to_csv(GT_CSV, index=False)

def show_video_frames(video_path, label="Frame"):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = st.slider(f"{label} Frame", 0, total - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption=f"{label} Frame {frame_id}", use_container_width=True)
    cap.release()

def show_top_k_matches(k=5, threshold=0.0):
    df = pd.read_csv(MATCH_CSV)
    df = df[df['Cosine Similarity'] >= threshold].sort_values(by="Cosine Similarity", ascending=False).head(k)
    st.subheader(f"Top {k} Matches (Similarity ≥ {threshold})")
    images = []
    for idx, row in df.iterrows():
        path = os.path.join(VISUALS_DIR, f'match_{idx:03d}.jpg')
        if os.path.exists(path):
            st.image(Image.open(path), caption=f"{row['Broadcast Image']} ↔ {row['Tacticam Match']} (Score: {row['Cosine Similarity']:.2f})", width=256)
            images.append(path)
    return df, images

def download_csv(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, filename, "text/csv")

def download_images_as_zip(paths, zip_name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in paths:
            if os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
    buf.seek(0)
    st.download_button("Download Images (ZIP)", buf, zip_name, "application/zip")

def compute_metrics(pred_df, gt_df):
    merged = pd.merge(pred_df, gt_df, on='Broadcast Image', suffixes=('_pred', '_true'))
    y_true = (merged['Tacticam Match_pred'] == merged['Tacticam Match_true']).astype(int)
    y_score = merged['Cosine Similarity']
    rank1 = y_true.sum() / len(y_true) if len(y_true) > 0 else 0.0
    map_score = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0
    mismatches = merged[y_true == 0]
    mismatches.to_csv(BAD_LOG_CSV, index=False)
    return rank1, map_score, merged, mismatches

def save_tp_fp_fn_visuals(merged_df, visuals_dir="outputs/match_visuals", save_dir="outputs/evaluation_visuals"):
    tp_dir = os.path.join(save_dir, "tp")
    fp_dir = os.path.join(save_dir, "fp")
    fn_dir = os.path.join(save_dir, "fn")
    for d in [tp_dir, fp_dir, fn_dir]:
        os.makedirs(d, exist_ok=True)

    for idx, row in merged_df.iterrows():
        pred = row["Tacticam Match_pred"]
        truth = row["Tacticam Match_true"]
        img_name = f"match_{idx:03d}.jpg"
        src_path = os.path.join(visuals_dir, img_name)
        if not os.path.exists(src_path):
            continue
        if pred == truth:
            shutil.copy(src_path, os.path.join(tp_dir, img_name))
        else:
            shutil.copy(src_path, os.path.join(fn_dir, img_name))

def show_conf_matrix(true_labels, pred_labels):
    labels = np.unique(np.concatenate((true_labels, pred_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    st.pyplot(fig)

def show_matched_view(df, mode='predicted'):
    st.subheader(f"Matched Pairs ({mode})")
    for idx, row in df.iterrows():
        b_img = row['Broadcast Image']
        t_img = row['Tacticam Match_pred'] if mode == 'predicted' else row['Tacticam Match_true']
        score = f"{row['Cosine Similarity']:.2f}" if mode == 'predicted' else "GT"
        vis_path = os.path.join(VISUALS_DIR, f"match_{idx:03d}.jpg")
        if os.path.exists(vis_path):
            st.image(Image.open(vis_path), caption=f"{b_img} ↔ {t_img} (Score: {score})", width=300)

def show_eval_visuals():
    st.subheader("Evaluation Visuals")
    eval_type = st.radio("Type", ['tp', 'fp', 'fn'], horizontal=True)
    e_dir = os.path.join(EVAL_VISUALS, eval_type)
    imgs = sorted([f for f in os.listdir(e_dir) if f.endswith('.jpg')]) if os.path.exists(e_dir) else []
    if imgs:
        idx = st.slider(f"{eval_type.upper()} Index", 0, len(imgs)-1)
        st.image(Image.open(os.path.join(e_dir, imgs[idx])), caption=f"{eval_type.upper()} Match {idx}", use_container_width=True)
    else:
        st.warning(f"No images in {e_dir}")

# App

st.set_page_config(page_title="Cross-Camera Player Dashboard", layout="wide")
st.title("Cross Camera Player Mapping")

# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K Matches", 1, 50, 10)
threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5)
video_opt = st.sidebar.selectbox("Video Viewer", list(VIDEO_PATHS.keys()))
show_video_frames(VIDEO_PATHS[video_opt], label=video_opt.capitalize())

# Display metrics from CSV
if os.path.exists(EVAL_METRICS):
    st.subheader("Evaluation Metrics (Raw CSV)")
    metrics_df = pd.read_csv(EVAL_METRICS)
    st.dataframe(metrics_df.style.format("{:.4f}"))
else:
    st.warning("⚠️ Metrics CSV not found!")

# Show top matches
st.markdown("---")
filtered_df, match_imgs = show_top_k_matches(k=top_k, threshold=threshold)
download_csv(filtered_df, "filtered_matches.csv")
download_images_as_zip(match_imgs, "matched_images.zip")

# Auto ground truth
auto_generate_ground_truth()

# Main Evaluation
if os.path.exists(GT_CSV):
    pred_df = pd.read_csv(MATCH_CSV)
    gt_df = pd.read_csv(GT_CSV)

    if not set(pred_df['Broadcast Image']) & set(gt_df['Broadcast Image']):
        gt_df = pred_df[['Broadcast Image', 'Tacticam Match']].copy()
        gt_df.to_csv(GT_CSV, index=False)

    rank1, map_score, merged, mismatches = compute_metrics(pred_df, gt_df)

    # Save TP/FP/FN Visuals
    save_tp_fp_fn_visuals(merged)

    st.markdown("##Evaluation Summary")
    st.metric("Rank@1 Accuracy", f"{rank1:.2%}")
    st.metric("Mean Average Precision (mAP)", f"{map_score:.4f}")

    st.subheader("Confusion Matrix")
    show_conf_matrix(merged['Tacticam Match_true'], merged['Tacticam Match_pred'])

    if not mismatches.empty:
        st.subheader("Mismatches")
        st.dataframe(mismatches[['Broadcast Image', 'Tacticam Match_pred', 'Tacticam Match_true', 'Cosine Similarity']])
    else:
        st.success("No mismatches found!")

    st.markdown("---")
    view_mode = st.radio("Matched Pair View", ["predicted", "ground_truth"], horizontal=True)
    show_matched_view(merged, mode=view_mode)

# Final: Evaluation visuals
st.markdown("---")
show_eval_visuals()
