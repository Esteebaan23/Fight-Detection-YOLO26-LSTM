from __future__ import annotations
import time
import cv2
import numpy as np

class RealtimeViewer:
    """
    backend:
      - "opencv": cv2.imshow (recommended for VSCode/local)
      - "colab": cv2_imshow + clear_output (not smooth but works)
    """
    def __init__(self, backend: str = "opencv", max_fps: float = 15.0, window_title: str = "Processing"):
        self.backend = backend
        self.max_fps = max_fps
        self.window_title = window_title
        self._last_t = 0.0

        self._colab_ready = False
        if backend == "colab":
            try:
                from google.colab.patches import cv2_imshow  # noqa
                from IPython.display import clear_output     # noqa
                self._colab_ready = True
            except Exception:
                self._colab_ready = False

    def show(self, frame_bgr: np.ndarray):
        now = time.time()
        if self.max_fps and self.max_fps > 0:
            if (now - self._last_t) < (1.0 / self.max_fps):
                return
        self._last_t = now

        if self.backend == "colab" and self._colab_ready:
            from google.colab.patches import cv2_imshow
            from IPython.display import clear_output
            clear_output(wait=True)
            cv2_imshow(frame_bgr)
        else:
            cv2.imshow(self.window_title, frame_bgr)
            cv2.waitKey(1)

    def close(self):
        if self.backend != "colab":
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass