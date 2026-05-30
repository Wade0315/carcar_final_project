import logging
import os
import time

import numpy as np
from picamera2 import Picamera2

from camera_base import CameraBase

logger = logging.getLogger(__name__)


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("logging initialized level=%s", logging.getLevelName(level))


class Camera(CameraBase):
    def __init__(
        self,
        width=320,
        height=240,
        flip_code=-1,
        model_path="best_ncnn_model",
        confidence=None,
        imgsz=None,
        target_class=None,
    ):
        super().__init__(width, height, flip_code)

        self.picam2 = Picamera2()
        self.closed = False
        self.model_path = os.getenv("YOLO_MODEL", model_path)
        self.confidence = float(os.getenv("YOLO_CONF", confidence or 0.25))
        self.imgsz = int(os.getenv("YOLO_IMGSZ", imgsz or max(self.width, self.height)))
        self.target_class = os.getenv("YOLO_CLASS", target_class or "").strip().lower()
        self.model = self.load_model(self.model_path)

        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)},
            controls={"FrameRate": 30},
            buffer_count=3,
        )
        self.picam2.configure(config)
        self.picam2.start()

        logger.info("camera activating...")
        time.sleep(2)
        self.lock_current_camera_controls()

    def load_model(self, model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for camera_YOLO.py. Install it before running this file."
            ) from exc

        logger.info("loading YOLO model from %s", model_path)
        return YOLO(model_path)

    def lock_current_camera_controls(self):
        metadata = self.picam2.capture_metadata()
        exposure_time = metadata.get("ExposureTime")
        analogue_gain = metadata.get("AnalogueGain")
        colour_gains = metadata.get("ColourGains")

        controls = {
            "AeEnable": False,
            "AwbEnable": False,
        }
        if exposure_time is not None:
            controls["ExposureTime"] = exposure_time
        if analogue_gain is not None:
            controls["AnalogueGain"] = analogue_gain
        if colour_gains is not None:
            controls["ColourGains"] = colour_gains

        self.picam2.set_controls(controls)
        logger.info(
            "lock camera ExposureTime=%s AnalogueGain=%s ColourGains=%s",
            exposure_time,
            analogue_gain,
            colour_gains,
        )

    def detect_frame(self, frame):
        floor_mask = self.build_floor_mask(frame)
        candidates = self.detect_yolo_candidates(frame)
        find_ball, error, target = self.choose_ball(candidates)
        return floor_mask, candidates, target, find_ball, error

    def detect_yolo_candidates(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", {}) or getattr(self.model, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

        candidates = []
        for box, conf, class_id in zip(xyxy, confidences, classes):
            class_name = str(names.get(int(class_id), class_id)).lower()
            if self.target_class and self.target_class not in class_name:
                continue

            x1, y1, x2, y2 = box
            x1 = int(max(0, min(self.width - 1, round(x1))))
            y1 = int(max(0, min(self.height - 1, round(y1))))
            x2 = int(max(0, min(self.width - 1, round(x2))))
            y2 = int(max(0, min(self.height - 1, round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1
            target_cx = x1 + w // 2
            target_cy = y1 + h // 2
            error_from_center = target_cx - self.width // 2
            ratio = min(w, h) / max(w, h)

            candidates.append({
                "rect": ((target_cx, target_cy), (w, h), 0),
                "bbox": (x1, y1, x2, y2),
                "area": w * h,
                "confidence": float(conf),
                "class_id": int(class_id),
                "class_name": class_name,
                "rect_cx": target_cx,
                "rect_cy": target_cy,
                "target_cx": target_cx,
                "target_cy": target_cy,
                "error": error_from_center,
                "w_h_ratio": ratio,
                "head_found": True,
                "source": "yolo",
            })

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return candidates

    def process_frame(self, frame):
        _, _, _, find_ball, error = self.detect_frame(frame)
        return find_ball, error

    def streaming(self):
        logger.info("Starting YOLO tracking...")
        at_frame = 0

        try:
            while True:
                raw_frame = self.picam2.capture_array()
                raw_frame = self.fix_orientation(raw_frame)

                if at_frame % 3 == 0:
                    find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error

                at_frame += 1

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.picam2.stop()
        self.closed = True
        logger.info("Camera closed.")


if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
