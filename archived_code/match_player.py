import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

# Paths
EMB_PATH = 'outputs/embeddings'

# Load embeddings
broadcast_embeddings = np.load(os.path.join(EMB_PATH, 'broadcast_embeddings.npy'))
tacticam_embeddings = np.load(os.path.join(EMB_PATH, 'tacticam_embeddings.npy'))

broadcast_filenames = np.load(os.path.join(EMB_PATH, 'broadcast_filenames.npy'), allow_pickle=True)
tacticam_filenames = np.load(os.path.join(EMB_PATH, 'tacticam_filenames.npy'), allow_pickle=True)

# Compute cosine similarity
similarity_matrix = cosine_similarity(broadcast_embeddings, tacticam_embeddings)

# Match broadcast → tacticam
matches = []
for i, broadcast_file in enumerate(broadcast_filenames):
    best_idx = np.argmax(similarity_matrix[i])
    best_match = tacticam_filenames[best_idx]
    sim_score = similarity_matrix[i, best_idx]
    matches.append((broadcast_file, best_match, float(sim_score)))

# Save results
os.makedirs('outputs', exist_ok=True)
with open('outputs/player_matches.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Broadcast Image', 'Tacticam Match', 'Cosine Similarity'])
    writer.writerows(matches)

# Print matches
print("\n🎯 Top matches:")
for i, (b, t, s) in enumerate(matches[:10]):
    print(f"[{i+1}] 🎥 {b} → 📹 {t} (Similarity: {s:.4f})")

print("\n✅ All matches saved to outputs/player_matches.csv")