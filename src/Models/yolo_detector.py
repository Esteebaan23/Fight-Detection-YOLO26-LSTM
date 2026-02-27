from __future__ import annotations
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, weights: str, names: tuple[str, ...], conf_thres: float):
        self.model = YOLO(weights)
        self.names = list(names)
        self.conf_thres = conf_thres

    def infer(self, frame_bgr: np.ndarray):
        # returns a single ultralytics result object
        return self.model(frame_bgr, conf=self.conf_thres, verbose=False)[0]