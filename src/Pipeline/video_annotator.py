from __future__ import annotations
from collections import deque
import cv2
import numpy as np
import torch

from src.config import PipelineConfig
from src.Models.yolo_detector import YOLODetector
from src.Models.lstm_classifier import LSTMTemporalClassifier
from src.Features.yolo_features import frame_features_from_yolo
from src.Viz.draw import draw_detections, put_global_label
from src.Viz.realtime_viewer import RealtimeViewer

class VideoAnnotator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = YOLODetector(cfg.yolo_weights, cfg.yolo_names, cfg.conf_thres)
        self.temporal = LSTMTemporalClassifier(cfg.lstm_weights, self.device, feat_dim=8)

        self.viewer = None
        if cfg.show_realtime:
            self.viewer = RealtimeViewer(cfg.display_backend, cfg.realtime_max_fps)

    def _open_writer(self, path: str, fps: float, W: int, H: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (W, H))

    def run(self, input_video: str) -> str:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cfg.fps_target is not None:
            step = max(1, int(round(src_fps / self.cfg.fps_target)))
            out_fps = float(self.cfg.fps_target)
        else:
            step = 1
            out_fps = float(src_fps)

        writer = self._open_writer(self.cfg.output_video, out_fps, W, H)

        feat_queue = deque(maxlen=self.cfg.window)
        current_label = "No Fight"

        frame_idx = 0
        kept = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step != 0:
                    frame_idx += 1
                    continue

                # YOLO
                res = self.detector.infer(frame)

                # features -> queue
                feat_queue.append(frame_features_from_yolo(res, W, H))

                # LSTM label when window full
                if len(feat_queue) == self.cfg.window:
                    window_feats = np.stack(feat_queue, axis=0)
                    current_label = self.temporal.predict_label(window_feats)

                out_frame = frame.copy()
                out_frame = draw_detections(out_frame, res, self.detector.names, topk=self.cfg.topk)
                out_frame = put_global_label(out_frame, current_label)

                if self.viewer is not None:
                    self.viewer.show(out_frame)

                writer.write(out_frame)

                kept += 1
                frame_idx += 1

                if self.cfg.max_frames is not None and kept >= self.cfg.max_frames:
                    break

        finally:
            cap.release()
            writer.release()
            if self.viewer is not None:
                self.viewer.close()

        return self.cfg.output_video