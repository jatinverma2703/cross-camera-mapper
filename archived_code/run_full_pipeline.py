import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import transforms
import torch
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Paths and Settings ------------------ #
VIDEO_PATHS = {
    'broadcast': 'data/broadcast.mp4',
    'tacticam': 'data/tacticam.mp4'
}
MODEL_PATH = 'models/yolo_ball_player.pt'
CROPS_DIR = 'outputs/detections'
EMBEDDINGS_DIR = 'outputs/embeddings'
VISUALS_DIR = 'outputs/match_visuals'
MATCH_CSV = 'outputs/player_matches.csv'

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

TARGET_CLASSES = [0, 1]  # player, goalkeeper
CONF_THRESHOLD = 0.5

# ------------------ Step 1: Detection ------------------- #
def detect_and_crop():
    model = YOLO(MODEL_PATH)
    for cam, path in VIDEO_PATHS.items():
        save_dir = os.path.join(CROPS_DIR, cam)
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Cannot open {path}")
            continue

        frame_id = crop_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD or cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                filename = f'{cam}_f{frame_id:04d}_p{crop_id}.jpg'
                cv2.imwrite(os.path.join(save_dir, filename), crop)
                crop_id += 1
            frame_id += 1
        cap.release()
    print("✅ Detection complete.\n")

# --------------- Step 2: Feature Extraction ------------- #
def extract_features():
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='osnet_x1_0_imagenet.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for cam in ['broadcast', 'tacticam']:
        image_dir = os.path.join(CROPS_DIR, cam)
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        embeddings = []
        valid_filenames = []

        for fname in tqdm(image_files, desc=f'Embedding: {cam}'):
            img_path = os.path.join(image_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                feat = extractor(img_tensor).cpu().numpy().flatten()
                embeddings.append(feat)
                valid_filenames.append(fname)
            except:
                continue

        np.save(os.path.join(EMBEDDINGS_DIR, f'{cam}_embeddings.npy'), np.array(embeddings))
        np.save(os.path.join(EMBEDDINGS_DIR, f'{cam}_filenames.npy'), np.array(valid_filenames))
    print("✅ Feature extraction done.\n")

# ------------------- Step 3: Matching ------------------- #
def match_players():
    emb1 = np.load(os.path.join(EMBEDDINGS_DIR, 'broadcast_embeddings.npy'))
    emb2 = np.load(os.path.join(EMBEDDINGS_DIR, 'tacticam_embeddings.npy'))
    names1 = np.load(os.path.join(EMBEDDINGS_DIR, 'broadcast_filenames.npy'))
    names2 = np.load(os.path.join(EMBEDDINGS_DIR, 'tacticam_filenames.npy'))

    sim_matrix = cosine_similarity(emb1, emb2)
    matches = []

    for i, row in enumerate(sim_matrix):
        best_idx = np.argmax(row)
        matches.append({
            'Broadcast Image': names1[i],
            'Tacticam Match': names2[best_idx],
            'Cosine Similarity': row[best_idx]
        })

    df = pd.DataFrame(matches)
    df.to_csv(MATCH_CSV, index=False)
    print("✅ Matching complete.\n")

# --------------- Step 4: Visualize Matches -------------- #
def visualize_matches():
    df = pd.read_csv(MATCH_CSV)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Visualizing'):
        path1 = os.path.join(CROPS_DIR, 'broadcast', row['Broadcast Image'])
        path2 = os.path.join(CROPS_DIR, 'tacticam', row['Tacticam Match'])

        if not os.path.exists(path1) or not os.path.exists(path2):
            continue

        try:
            img1 = Image.open(path1).resize((128, 256))
            img2 = Image.open(path2).resize((128, 256))
            combo = Image.new('RGB', (256, 256))
            combo.paste(img1, (0, 0))
            combo.paste(img2, (128, 0))
            combo.save(os.path.join(VISUALS_DIR, f'match_{idx:03d}.jpg'))
        except:
            continue
    print("✅ Visual match images saved.\n")

# ---------------------- Run All ------------------------ #
if __name__ == "__main__":
    detect_and_crop()
    extract_features()
    match_players()
    visualize_matches()