from __future__ import annotations
import cv2
import numpy as np

def draw_detections(
    frame_bgr,
    result,
    names,
    topk=50,
    allowed_classes=None   
):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame_bgr

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy().astype(float)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)

    order = np.argsort(-conf)[:topk]

    for i in order:
        c = cls[i]

        # 🔹 Filtrar clases si se especifica
        if allowed_classes is not None and c not in allowed_classes:
            continue

        x1, y1, x2, y2 = xyxy[i]
        p = conf[i]
        label = f"{names[c]} {p:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return frame_bgr

def put_global_label(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    color = (0, 0, 255) if "Fight" in label else (0, 255, 0)
    cv2.putText(frame_bgr, label, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame_bgr