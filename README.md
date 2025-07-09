# 🎥 Cross Camera Mappig

This project is a **Streamlit-based interactive dashboard** for cross-camera player re-identification in sports videos. It allows you to visualize matched detections from multiple camera angles, evaluate performance metrics, and inspect mismatches — all in a user-friendly interface.

---

## 🚀 Features

- 📊 Evaluation Metrics: Rank@1 Accuracy, mAP, Confusion Matrix
- 🖼️ Match Viewer: Visualize top-k matches between broadcast and tacticam frames
- ❌ Mismatch Inspection: Browse false positives and negatives
- 🎬 Frame Explorer: Inspect individual frames from videos
- 📥 Export Tools: Download filtered results as CSV or ZIP
- 📂 Visual Debugger: See categorized TP / FP / FN samples

---

## 🗂️ Folder Structure
cross-camera-reid-dashboard/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── data/
│ ├── broadcast.mp4 # Broadcast camera video
│ └── tacticam.mp4 # Tacticam video
├── outputs/
│ ├── player_matches.csv # Predicted image pairs + similarity
│ ├── ground_truth_matches.csv # Auto-generated GT if not provided
│ ├── match_visuals/ # Visuals of matched images
│ ├── evaluation_visuals/ # TP / FP / FN image categories
│ └── evaluation_metrics.csv # Evaluation summary
└── archived_code/ # Old scripts/models (optional, ignored if large)


## ⚙️ Installation

```bash
git clone https://github.com/jatinverma2703/cross-camera-reid-dashboard.git
cd cross-camera-reid-dashboard
pip install -r requirements.txt


## 🖥️ Run Locally
streamlit run streamlit_dashboard.py

☁️ Deploy to Streamlit Cloud
1.Visit streamlit.io/cloud
2.Click New app
3.Select this repo and set:
4.Main file: app.py
5.Click Deploy

📦 Requirements
Your requirements.txt should contain:
streamlit
opencv-python
pandas
numpy
Pillow
scikit-learn
seaborn
matplotlib

👤 Author
Jatin Verma
GitHub: @jatinverma2703

📄 License
This project is licensed under the MIT License.



