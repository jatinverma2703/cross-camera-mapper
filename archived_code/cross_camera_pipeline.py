import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

# Config
VIDEO_PATHS = {
    'broadcast': 'data/broadcast.mp4',
    'tacticam': 'data/tacticam.mp4'
}
MODEL_PATH = 'models/yolo_ball_player.pt'
DETECTIONS_DIR = 'outputs/detections'
EMBEDDINGS_DIR = 'outputs/embeddings'
MATCH_VIS_DIR = 'outputs/match_visuals'
EVAL_VIS_DIR = 'outputs/evaluation_visuals'
MATCH_CSV = 'outputs/player_matches.csv'
GROUND_TRUTH_CSV = 'outputs/ground_truth_matches.csv'
METRICS_REPORT_CSV = 'outputs/evaluation_metrics.csv'

# Ensure necessary directories
os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MATCH_VIS_DIR, exist_ok=True)
os.makedirs(EVAL_VIS_DIR, exist_ok=True)

# Step 1: Detection
def run_detection():
    model = YOLO(MODEL_PATH)
    for cam, path in VIDEO_PATHS.items():
        save_dir = os.path.join(DETECTIONS_DIR, cam)
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {path}")
            continue
        frame_id, crop_id = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = model(frame)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5: continue
                if cls_id in [0, 1]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    filename = f"{cam}_f{frame_id:04d}_p{crop_id}.jpg"
                    cv2.imwrite(os.path.join(save_dir, filename), crop)
                    crop_id += 1
            frame_id += 1
        cap.release()

# Step 2: Feature Extraction
def run_feature_extraction():
    extractor = FeatureExtractor(model_name='osnet_x1_0',
                                 model_path='osnet_x1_0_imagenet.pth',
                                 device='cpu')
    for cam in ['broadcast', 'tacticam']:
        folder = os.path.join(DETECTIONS_DIR, cam)
        images = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
        full_paths = [os.path.join(folder, img) for img in images]
        features = extractor(full_paths)
        np.save(os.path.join(EMBEDDINGS_DIR, f"{cam}_embeddings.npy"), features)
        np.save(os.path.join(EMBEDDINGS_DIR, f"{cam}_filenames.npy"), np.array(images))

# Step 3: Matching
def run_matching():
    A = np.load(os.path.join(EMBEDDINGS_DIR, 'broadcast_embeddings.npy'))
    B = np.load(os.path.join(EMBEDDINGS_DIR, 'tacticam_embeddings.npy'))
    names_A = np.load(os.path.join(EMBEDDINGS_DIR, 'broadcast_filenames.npy'))
    names_B = np.load(os.path.join(EMBEDDINGS_DIR, 'tacticam_filenames.npy'))
    sim = cosine_similarity(A, B)
    top_matches = sim.argmax(axis=1)
    scores = sim.max(axis=1)
    match_data = []
    for i, (idx, score) in enumerate(zip(top_matches, scores)):
        match_data.append({
            'Broadcast Image': names_A[i],
            'Tacticam Match': names_B[idx],
            'Cosine Similarity': round(score, 4)
        })
    pd.DataFrame(match_data).to_csv(MATCH_CSV, index=False)

# Step 4: Visualization
def run_visualization():
    df = pd.read_csv(MATCH_CSV)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img1_path = os.path.join(DETECTIONS_DIR, 'broadcast', row['Broadcast Image'])
        img2_path = os.path.join(DETECTIONS_DIR, 'tacticam', row['Tacticam Match'])
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            continue
        img1 = Image.open(img1_path).resize((128, 256))
        img2 = Image.open(img2_path).resize((128, 256))
        combined = Image.new('RGB', (256, 256))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (128, 0))
        combined.save(os.path.join(MATCH_VIS_DIR, f'match_{idx:03d}.jpg'))

# Step 5: Evaluation
def run_evaluation():
    if not os.path.exists(GROUND_TRUTH_CSV):
        print("⚠️ No ground_truth_matches.csv found. Skipping evaluation.")
        return

    pred_df = pd.read_csv(MATCH_CSV)
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)

    pred_df.columns = pred_df.columns.str.lower().str.strip().str.replace(" ", "_")
    gt_df.columns = gt_df.columns.str.lower().str.strip().str.replace(" ", "_")

    pred_set = set(tuple(x) for x in pred_df[['broadcast_image', 'tacticam_match']].values)
    gt_set = set(tuple(x) for x in gt_df[['broadcast_image', 'tacticam_match']].values)

    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set

    precision = len(tp) / (len(tp) + len(fp) + 1e-6)
    recall = len(tp) / (len(tp) + len(fn) + 1e-6)

    # Rank@1 = correct top prediction / total
    rank1_count = sum(1 for row in pred_df.itertuples()
                      if (row.broadcast_image, row.tacticam_match) in gt_set)
    rank1 = rank1_count / len(pred_df)

    # mAP (simplified, assuming one GT match per broadcast)
    ap_list = []
    for gt_row in gt_df.itertuples():
        b_img = gt_row.broadcast_image
        if b_img in pred_df['broadcast_image'].values:
            pred_row = pred_df[pred_df['broadcast_image'] == b_img].iloc[0]
            if (pred_row.broadcast_image, pred_row.tacticam_match) in gt_set:
                ap_list.append(1.0)
            else:
                ap_list.append(0.0)
    mAP = sum(ap_list) / len(ap_list) if ap_list else 0.0

    print(f"\n📊 Evaluation Report")
    print(f"✔️ TP: {len(tp)} | ❌ FP: {len(fp)} | ⚠️ FN: {len(fn)}")
    print(f"🎯 Precision: {precision:.4f} | 📈 Recall: {recall:.4f}")
    print(f"🏆 Rank@1: {rank1:.4f} | 🧠 mAP: {mAP:.4f}")

    # Save metrics to file
    pd.DataFrame([{
        'TP': len(tp), 'FP': len(fp), 'FN': len(fn),
        'Precision': precision,
        'Recall': recall,
        'Rank@1': rank1,
        'mAP': mAP
    }]).to_csv(METRICS_REPORT_CSV, index=False)

    # Visualize TP/FP/FN
    for label, pair_set in zip(['tp', 'fp', 'fn'], [tp, fp, fn]):
        out_dir = os.path.join(EVAL_VIS_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        for i, (b_img, t_img) in enumerate(pair_set):
            path1 = os.path.join(DETECTIONS_DIR, 'broadcast', b_img)
            path2 = os.path.join(DETECTIONS_DIR, 'tacticam', t_img)
            if not os.path.exists(path1) or not os.path.exists(path2):
                continue
            img1 = Image.open(path1).resize((128, 256))
            img2 = Image.open(path2).resize((128, 256))
            combined = Image.new('RGB', (256, 256))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (128, 0))
            combined.save(os.path.join(out_dir, f'{label}_{i:03d}.jpg'))

# Main Execution
if __name__ == "__main__":
    run_detection()
    run_feature_extraction()
    run_matching()
    run_visualization()
    run_evaluation()
    print("\n✅ All steps completed including evaluation and metrics report.")