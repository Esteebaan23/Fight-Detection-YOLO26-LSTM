from __future__ import annotations
import argparse
import os

from src.config import PipelineConfig
from src.Pipeline.video_annotator import VideoAnnotator

def parse_args():
    ap = argparse.ArgumentParser(description="Annotate a video with YOLO detections + LSTM fight classification.")
    ap.add_argument("--input", required=True, help="Path to input video (e.g. videos/V_997.mp4)")
    ap.add_argument("--yolo-weights", default="weights/best.pt", help="Path to YOLO weights")
    ap.add_argument("--lstm-weights", default="weights/lstm_fight.pt", help="Path to LSTM weights")
    ap.add_argument("--output", default="outputs/output_annotated.mp4", help="Path to output mp4")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    ap.add_argument("--window", type=int, default=32, help="Temporal window length")
    ap.add_argument("--fps-target", type=float, default=None, help="Optional output FPS (downsample)")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional limit on processed frames")
    ap.add_argument("--no-realtime", action="store_true", help="Disable realtime display")
    ap.add_argument("--display", default="opencv", choices=["opencv", "colab"], help="Realtime display backend")
    ap.add_argument("--display-fps", type=float, default=15.0, help="Max FPS for realtime display")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    cfg = PipelineConfig(
        yolo_weights=args.yolo_weights,
        lstm_weights=args.lstm_weights,
        conf_thres=args.conf,
        window=args.window,
        fps_target=args.fps_target,
        max_frames=args.max_frames,
        output_video=args.output,
        show_realtime=(not args.no_realtime),
        display_backend=args.display,
        realtime_max_fps=args.display_fps,
    )

    annotator = VideoAnnotator(cfg)
    out_path = annotator.run(args.input)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()