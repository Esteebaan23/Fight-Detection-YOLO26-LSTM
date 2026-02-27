from __future__ import annotations
import numpy as np

def frame_features_from_yolo(result, img_w: int, img_h: int) -> np.ndarray:
    """
    8D feature vector:
    [viol_count, viol_sumconf, viol_area, viol_maxconf,
     weap_count, weap_sumconf, weap_area, weap_maxconf]
    """
    v_count = v_sumconf = v_area = v_maxconf = 0.0
    w_count = w_sumconf = w_area = w_maxconf = 0.0

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((8,), dtype=np.float32)

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy().astype(float)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)

    denom = (img_w * img_h + 1e-9)
    for c, p, (x1, y1, x2, y2) in zip(cls, conf, xyxy):
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / denom
        if c == 0:
            v_count += 1
            v_sumconf += p
            v_area += area
            v_maxconf = max(v_maxconf, p)
        elif c == 1:
            w_count += 1
            w_sumconf += p
            w_area += area
            w_maxconf = max(w_maxconf, p)

    return np.array([v_count, v_sumconf, v_area, v_maxconf,
                     w_count, w_sumconf, w_area, w_maxconf], dtype=np.float32)