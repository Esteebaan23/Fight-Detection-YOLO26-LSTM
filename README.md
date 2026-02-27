Fight Detection Pipeline (YOLO26 + LSTM)

A modular computer vision system for detecting violent events in video using:

- **YOLO** for object detection (e.g., violence, weapon)
- **Temporal LSTM with Attention** for sequence-level fight classification
- Real-time visualization + annotated video export

---

## 🔎 Overview

This pipeline performs:

1. Frame-level object detection using YOLO
2. Extraction of an 8D feature vector per frame:
   - Violence count
   - Violence confidence sum
   - Violence relative area
   - Violence max confidence
   - Weapon count
   - Weapon confidence sum
   - Weapon relative area
   - Weapon max confidence
3. Temporal classification using a BiLSTM with attention
4. Video annotation with:
   - Bounding boxes
   - readme_content = """# Fight Detection Pipeline (YOLO + LSTM)

A modular computer vision system for detecting violent events in video using:

- **YOLO** for object detection (e.g., violence, weapon)
- **Temporal LSTM with Attention** for sequence-level fight classification
- Real-time visualization + annotated video export

---

## 🔎 Overview

This pipeline performs:

1. Frame-level object detection using YOLO
2. Extraction of an 8D feature vector per frame:
   - Violence count
   - Violence confidence sum
   - Violence relative area
   - Violence max confidence
   - Weapon count
   - Weapon confidence sum
   - Weapon relative area
   - Weapon max confidence
3. Temporal classification using a BiLSTM with attention
4. Video annotation with:
   - Bounding boxes
   - Global label: **Fight / No Fight**
   - Optional real-time display

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




