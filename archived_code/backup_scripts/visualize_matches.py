import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Paths
MATCH_CSV = 'outputs/player_matches.csv'
DETECTIONS_DIR = 'outputs/detections'
OUTPUT_DIR = 'outputs/match_visuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load matches
df = pd.read_csv(MATCH_CSV)

# Normalize column names for easier access
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

for idx, row in tqdm(df.iterrows(), total=len(df), desc="📸 Creating visual match pairs"):
    broadcast_img = os.path.join(DETECTIONS_DIR, 'broadcast', row['broadcast_image'])
    tacticam_img = os.path.join(DETECTIONS_DIR, 'tacticam', row['tacticam_match'])

    if not os.path.exists(broadcast_img) or not os.path.exists(tacticam_img):
        print(f"⚠️ Missing image: {broadcast_img} or {tacticam_img}")
        continue

    try:
        img1 = Image.open(broadcast_img).resize((128, 256))
        img2 = Image.open(tacticam_img).resize((128, 256))
        combined = Image.new('RGB', (256, 256))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (128, 0))

        save_path = os.path.join(OUTPUT_DIR, f'match_{idx:03d}.jpg')
        combined.save(save_path)
    except Exception as e:
        print(f"❌ Error processing row {idx}: {e}")

print(f"\n✅ All visual pairs saved to: {OUTPUT_DIR}")