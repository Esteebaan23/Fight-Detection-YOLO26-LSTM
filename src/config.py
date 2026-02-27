from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PipelineConfig:
    # Paths
    yolo_weights: str
    lstm_weights: str

    # Model / labels
    yolo_names: Tuple[str, ...] = ("violence", "weapon")

    # Inference params
    conf_thres: float = 0.5
    window: int = 32
    fps_target: Optional[float] = None
    max_frames: Optional[int] = None
    topk: int = 50

    # IO
    output_video: str = "outputs/output_annotated.mp4"

    # Realtime display
    show_realtime: bool = True
    realtime_max_fps: float = 15.0
    display_backend: str = "opencv"  # "opencv" or "colab"