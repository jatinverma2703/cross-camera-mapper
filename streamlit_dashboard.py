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

# Paths
MATCH_CSV = 'outputs/player_matches.csv'
VISUALS_DIR = 'outputs/match_visuals'
DETECTIONS_DIR = 'outputs/detections'
GT_CSV = 'outputs/ground_truth_matches.csv'
BAD_LOG_CSV = 'outputs/bad_predictions_log.csv'
VIDEO_PATHS = {
    'broadcast': 'data/broadcast.mp4',
    'tacticam': 'data/tacticam.mp4'
}

# Auto-generate ground truth file
def auto_generate_ground_truth():
    if os.path.exists(GT_CSV) and os.path.getsize(GT_CSV) > 0:
        return
    broadcast_dir = os.path.join(DETECTIONS_DIR, 'broadcast')
    tacticam_dir = os.path.join(DETECTIONS_DIR, 'tacticam')
    broadcast_images = sorted([f for f in os.listdir(broadcast_dir) if f.endswith('.jpg')])
    tacticam_images = sorted([f for f in os.listdir(tacticam_dir) if f.endswith('.jpg')])
    n = min(len(broadcast_images), len(tacticam_images))
    df = pd.DataFrame(list(zip(broadcast_images[:n], tacticam_images[:n])),
                      columns=['Broadcast Image', 'Tacticam Match'])
    df.to_csv(GT_CSV, index=False)

# Show video frames
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

# Show top-k matches
def show_top_k_matches(csv_path, visuals_dir, k=5, threshold=0.0):
    df = pd.read_csv(csv_path)
    df = df[df['Cosine Similarity'] >= threshold]
    df = df.sort_values(by="Cosine Similarity", ascending=False).head(k)
    st.subheader(f"Top {k} Matches (Similarity ≥ {threshold})")
    img_paths = []
    for idx, row in df.iterrows():
        match_img = os.path.join(visuals_dir, f'match_{idx:03d}.jpg')
        if os.path.exists(match_img):
            st.image(Image.open(match_img), caption=f"{row['Broadcast Image']} ↔ {row['Tacticam Match']} (Score: {row['Cosine Similarity']:.2f})", width=256)
            img_paths.append(match_img)
    return df, img_paths

# Download CSV
def download_filtered_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered CSV", csv, "filtered_matches.csv", "text/csv")

# Download ZIP
def download_matched_images_as_zip(image_paths):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path in image_paths:
            if os.path.exists(path):
                zf.write(path, arcname=os.path.basename(path))
    buffer.seek(0)
    st.download_button("Download Matched Images (ZIP)", buffer, "matched_images.zip", "application/zip")

# Compute metrics and mismatches
def compute_metrics(pred_df, gt_df):
    merged = pd.merge(pred_df, gt_df, on='Broadcast Image', suffixes=('_pred', '_true'))
    y_true = (merged['Tacticam Match_pred'] == merged['Tacticam Match_true']).astype(int)
    y_score = merged['Cosine Similarity']
    rank1 = y_true.sum() / len(y_true) if len(y_true) > 0 else 0.0
    map_score = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0
    mismatches = merged[y_true == 0]
    mismatches.to_csv(BAD_LOG_CSV, index=False)
    return rank1, map_score, y_true, (merged['Tacticam Match_true'], merged['Tacticam Match_pred']), mismatches, merged

# Show confusion matrix
def show_confusion_matrix(true_labels, pred_labels):
    labels = np.unique(np.concatenate((true_labels, pred_labels)))
    if len(labels) == 0:
        st.warning("No labels available for confusion matrix.")
        return
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Mismatch table
def show_mismatch_table(mismatches):
    st.markdown("Incorrect Matches")
    if mismatches.empty:
        st.success("No mismatches found.")
    else:
        st.dataframe(mismatches[['Broadcast Image', 'Tacticam Match_pred', 'Tacticam Match_true', 'Cosine Similarity']])

# Match viewer
def show_toggle_matched_pairs(df, visuals_dir, mode="predicted"):
    st.markdown(f"Matched Pairs Viewer ({mode.capitalize()})")
    for idx, row in df.iterrows():
        broadcast_img = row['Broadcast Image']
        if mode == "predicted":
            tacticam_img = row['Tacticam Match_pred']
            score = row['Cosine Similarity']
        else:
            tacticam_img = row['Tacticam Match_true']
            score = "Ground Truth"
        visual_path = os.path.join(visuals_dir, f"match_{idx:03d}.jpg")
        if os.path.exists(visual_path):
            st.image(Image.open(visual_path), caption=f"{broadcast_img} ↔ {tacticam_img} (Score: {score})", width=300)

# Colored metric display
def colored_metric(label, value, threshold_good, threshold_ok, fmt="{:.2%}"):
    if value >= threshold_good:
        color = "green"
    elif value >= threshold_ok:
        color = "orange"
    else:
        color = "red"
    st.markdown(f"**{label}:** <span style='color:{color}; font-weight:bold'>{fmt.format(value)}</span>", unsafe_allow_html=True)

# Main app
def main():
    st.set_page_config(page_title="Cross Camera Mapper", layout="wide")
    st.title("Cross Camera Mapper")

    st.sidebar.header("Filter and Settings")
    top_k = st.sidebar.slider("Top-K Matches", 1, 50, 10)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5)

    st.sidebar.header("Video Frame Viewer")
    video_choice = st.sidebar.selectbox("Select Video", list(VIDEO_PATHS.keys()))
    if video_choice:
        show_video_frames(VIDEO_PATHS[video_choice], label=video_choice.capitalize())

    st.markdown("---")
    df_filtered, matched_imgs = show_top_k_matches(MATCH_CSV, VISUALS_DIR, k=top_k, threshold=similarity_threshold)

    st.markdown("Export Results")
    download_filtered_csv(df_filtered)
    download_matched_images_as_zip(matched_imgs)

    auto_generate_ground_truth()

    if os.path.exists(GT_CSV):
        st.markdown("---")
        st.markdown("Evaluation Metrics")

        try:
            pred_df = pd.read_csv(MATCH_CSV)
            gt_df = pd.read_csv(GT_CSV)

            common = set(pred_df['Broadcast Image']) & set(gt_df['Broadcast Image'])
            if len(common) == 0:
                st.warning("No common Broadcast Images found between predictions and ground truth. Using predicted data as dummy GT.")
                gt_df = pred_df[['Broadcast Image', 'Tacticam Match']].copy()
                gt_df.to_csv(GT_CSV, index=False)

            rank1, map_score, y_true, (true_labels, pred_labels), mismatches, merged_all = compute_metrics(pred_df, gt_df)

            col1, col2 = st.columns(2)
            with col1:
                colored_metric("Rank@1 Accuracy", rank1, 0.8, 0.5)
            with col2:
                colored_metric("Mean Average Precision", map_score, 0.7, 0.4, fmt="{:.4f}")

            st.markdown("Confusion Matrix")
            show_confusion_matrix(true_labels, pred_labels)

            show_mismatch_table(mismatches)

            st.markdown("Toggle View of Matched Pairs")
            view_mode = st.radio("View Mode", options=["predicted", "ground_truth"], horizontal=True)
            show_toggle_matched_pairs(merged_all, VISUALS_DIR, mode=view_mode)

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
    else:
        st.warning("Ground truth file not found.")

if __name__ == "__main__":
    main()