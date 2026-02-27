import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import torch.nn.functional as F

# -------- Load models --------
YOLO_WEIGHTS = "best.pt"  
LSTM_WEIGHTS = "lstm_fight.pt"                      

yolo = YOLO(YOLO_WEIGHTS)

# Your YOLO class names (ensure order matches training)
YOLO_NAMES = ["violence", "weapon"]  # class 0, class 1

# LSTM model definition
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Capa para calcular la importancia de cada frame
        self.attention = nn.Linear(hidden_size * 2, 1) # *2 porque es bidireccional

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size * 2]

        # Calculamos los pesos de atención para cada frame
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # [batch_size, seq_len, 1]

        # Multiplicamos la salida del LSTM por sus pesos y sumamos
        context = torch.sum(attn_weights * lstm_out, dim=1) # [batch_size, hidden_size * 2]
        return context, attn_weights

class UltimateLSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden=256, num_layers=3, dropout=0.5):
        super().__init__()

        # 1. LSTM Extra Profundo (3 capas de memoria)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 2. Nuestro nuevo "Cerebro" de Atención
        self.attention = TemporalAttention(hidden)

        # 3. Cabeza Clasificadora de Alto Rendimiento
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.LayerNorm(512),  # LayerNorm es más estable que BatchNorm para secuencias
            nn.GELU(),          # GELU es una activación más moderna y suave que ReLU
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Pasamos por el LSTM
        out, _ = self.lstm(x)

        # En lugar de tomar el último frame, usamos Atención para extraer lo mejor de toda la secuencia
        context, _ = self.attention(out)

        # Clasificamos
        return self.head(context)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
lstm = UltimateLSTMClassifier(feat_dim=8).to(device)
lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=device))
lstm.eval()

def frame_features_from_yolo(result, img_w, img_h):
    """
    Build an 8D feature vector from YOLO detections for temporal classification.
    """
    # [viol_count, viol_sumconf, viol_area, viol_maxconf, weap_count, weap_sumconf, weap_area, weap_maxconf]
    v_count = v_sumconf = v_area = v_maxconf = 0.0
    w_count = w_sumconf = w_area = w_maxconf = 0.0

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.array([0,0,0,0, 0,0,0,0], dtype=np.float32)

    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy().astype(float)
    xyxy = boxes.xyxy.cpu().numpy().astype(float)

    for c, p, (x1,y1,x2,y2) in zip(cls, conf, xyxy):
        area = max(0.0, (x2-x1)) * max(0.0, (y2-y1)) / (img_w * img_h + 1e-9)
        if c == 0:  # violence
            v_count += 1; v_sumconf += p; v_area += area; v_maxconf = max(v_maxconf, p)
        elif c == 1:  # weapon
            w_count += 1; w_sumconf += p; w_area += area; w_maxconf = max(w_maxconf, p)

    return np.array([v_count, v_sumconf, v_area, v_maxconf,
                     w_count, w_sumconf, w_area, w_maxconf], dtype=np.float32)

def lstm_label_from_window(window_feats):
    """
    window_feats: np.array [T, 8]
    Returns a hard label: "FIGHT" or "NO FIGHT"
    """
    xb = torch.tensor(window_feats[None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = lstm(xb)
        pred = int(torch.argmax(logits, dim=1).item())
    return "FIGHT" if pred == 1 else "NO FIGHT"

def draw_detections(frame, result, topk=20):
    """
    Draw YOLO detections on a frame.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy().astype(float)
    xyxy = boxes.xyxy.cpu().numpy().astype(int)

    # Sort by confidence
    order = np.argsort(-conf)[:topk]

    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        c = cls[i]
        p = conf[i]
        label = f"{YOLO_NAMES[c]} {p:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return frame

def annotate_video_with_yolo_lstm(
    input_video,
    output_video="/content/output_annotated.mp4",
    window=32,
    fps_target=None,  
    conf_thres=0.85,
    max_frames=None
):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional downsample to fps_target
    if fps_target is not None:
        step = max(1, int(round(src_fps / fps_target)))
        out_fps = fps_target
    else:
        step = 1
        out_fps = src_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, out_fps, (W, H))

    feat_queue = deque(maxlen=window)
    frame_idx = 0
    kept = 0
    current_label = "NORMAL (No Fight)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue
        # YOLO inference
        res = yolo(frame, conf=conf_thres, verbose=False)[0]
        # Update temporal features
        feat_queue.append(frame_features_from_yolo(res, W, H))

        # Once window is full, update label
        if len(feat_queue) == window:
            window_feats = np.stack(list(feat_queue), axis=0)

            # 1. El LSTM nos dice si hay pelea o no
            lstm_pred = lstm_label_from_window(window_feats)

            # 2. NUEVA LÓGICA: Preguntamos a YOLO si vio un arma
            if lstm_pred == "FIGHT":
                current_label = "Fight"
            else:
                current_label = "No Fight"

        # Draw boxes + global label
        out_frame = frame.copy()
        out_frame = draw_detections(out_frame, res)
        color_texto = (0, 0, 255) if "Fight" in current_label else (0, 255, 0)

        # Put global label on the frame (borde blanco para que resalte)
        #cv2.putText(out_frame, f" {current_label}", (15, 30),
         #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out_frame, f"{current_label}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2, cv2.LINE_AA)

        writer.write(out_frame)
        kept += 1
        frame_idx += 1

        if max_frames is not None and kept >= max_frames:
            break

    cap.release()
    writer.release()
    return output_video

# --- EJECUCIÓN ---
out_path = annotate_video_with_yolo_lstm("V_997.mp4", output_video="output_annotated2.mp4", 
            conf_thres = 0.5)
print("¡Listo! Video guardado en:", out_path)

