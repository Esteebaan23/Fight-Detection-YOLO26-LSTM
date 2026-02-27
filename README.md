## Fight Detection Pipeline (YOLO26 + LSTM)

A modular computer vision system for detecting violent events in video using:

- **YOLO** for object detection (e.g., violence, weapon)
- **Temporal LSTM with Attention** for sequence-level fight classification
- Real-time visualization + annotated video export

<p align="center">
  <img src="Images/Demo.png" alt="Pipeline Overview" width="700"/>
</p>
---

## 🔎 Overview

This pipeline performs:

1. Frame-level object detection using YOLO
2. Extraction of an 8D feature vector per frame:
   - Violence count
   - Violence confidence sum
   - Violence relative area
   - Violence max confidence
3. Temporal classification using a BiLSTM with attention
4. Video annotation with:
   - Bounding boxes
   - readme_content = """# Fight Detection Pipeline (YOLO + LSTM)

A modular computer vision system for detecting violent events in video using:

- **YOLO** for object detection (e.g., violence, weapon)
- **Temporal LSTM with Attention** for sequence-level fight classification
- Real-time visualization + annotated video export

---
## 📁 Project Structure
``` bash
fight-detector-yolo-lstm/
│
├── src/
│ ├── config.py
│ ├── models/
│ │ ├── yolo_detector.py
│ │ └── lstm_classifier.py
│ ├── features/
│ │ └── yolo_features.py
│ ├── viz/
│ │ ├── draw.py
│ │ └── realtime_viewer.py
│ └── pipeline/
│ └── video_annotator.py
│
├── scripts/
│ └── annotate_video.py
│
├── weights/
├── videos/
├── outputs/
├── requirements.txt
└── README.md
```
---
⚙️ Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\\Scripts\\activate       # Windows
```


Install dependencies:
```bash
pip install -r requirements.txt
```

🚀 Run the Pipeline

Basic usage:
```bash
python -m scripts.annotate_video \
    --input videos/V_997.mp4 \
    --output outputs/output.mp4 \
    --conf 0.5 \
    --window 32
```

| Argument        | Description                      |
| --------------- | -------------------------------- |
| `--conf`        | YOLO confidence threshold        |
| `--window`      | Temporal window size for LSTM    |
| `--fps-target`  | Downsample FPS for processing    |
| `--max-frames`  | Limit number of processed frames |
| `--display`     | `opencv` or `colab`              |
| `--display-fps` | Max FPS for realtime display     |

---
## LSTM Classifier
- 3-layer BiLSTM
- Attention mechanism
- LayerNorm + GELU head
- Binary classification (Fight / No Fight)

---
## 📊 Output

The system produces:

- Annotated MP4 video
- Bounding boxes per detection

---
## ⚠ Notes

- GPU recommended for real-time performance.
- Weights must match the architecture defined in lstm_classifier.py.

---
## 👤 Author

Harold Lucero N.
