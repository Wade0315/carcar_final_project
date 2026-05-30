import logging
import os
import time

import cv2
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
        yolo_mask = self.build_yolo_mask(frame, candidates)
        return floor_mask, yolo_mask, candidates, target, find_ball, error

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

    def build_yolo_mask(self, frame, candidates):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["bbox"]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def process_frame(self, frame):
        floor_mask, yolo_mask, candidates, target, find_ball, error = self.detect_frame(frame)
        self.visualize_frame(frame, floor_mask, candidates, target)
        return frame, floor_mask, yolo_mask, find_ball, error

    def build_display_mask(self, floor_mask, yolo_mask):
        return yolo_mask

    def draw_target(self, frame, target):
        target_cx = target["target_cx"]
        target_cy = target["target_cy"]

        cv2.circle(frame, (target_cx, target_cy), 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            "TARGET",
            (target_cx - 25, target_cy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.line(
            frame,
            (self.width // 2, target_cy),
            (target_cx, target_cy),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"error={target['error']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{target['class_name']} {target['confidence']:.2f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    def visualize_frame(self, frame, floor_mask, candidates, target):
        self.draw_center_line(frame)
        self.draw_floor_boundary(frame, floor_mask)
        self.draw_candidates(frame, candidates)
        if target is not None:
            self.draw_target(frame, target)

    def draw_center_line(self, frame):
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (0, 255, 255), 1)

    def draw_floor_boundary(self, frame, floor_mask):
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 255), 2)

    def draw_candidates(self, frame, candidates):
        for ball in candidates:
            x1, y1, x2, y2 = ball["bbox"]
            target_cx = ball["target_cx"]
            target_cy = ball["target_cy"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (target_cx, target_cy), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{ball['class_name']} {ball['confidence']:.2f}",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

    def streaming(self):
        logger.info("Starting YOLO tracking... Press 'q' to quit.")
        at_frame = 0
        processed_frame = None
        floor_mask = None
        yolo_mask = None

        try:
            while True:
                raw_frame = self.picam2.capture_array()
                raw_frame = self.fix_orientation(raw_frame)

                if at_frame % 3 == 0:
                    processed_frame, floor_mask, yolo_mask, find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error

                if processed_frame is not None:
                    cv2.imshow("Robot View", processed_frame)

                if floor_mask is not None and yolo_mask is not None:
                    cv2.imshow("YOLO Mask", self.build_display_mask(floor_mask, yolo_mask))

                at_frame += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def single_test(self, output_dir="/home/waryt/Desktop/image", filename=None):
        logger.info("capturing photo...")
        os.makedirs(output_dir, exist_ok=True)

        raw_frame = self.picam2.capture_array()
        raw_frame = self.fix_orientation(raw_frame)

        processed_frame, floor_mask, yolo_mask, find_ball, error = self.process_frame(raw_frame)
        logger.info("find_ball=%s error=%s", find_ball, error)

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"yolo_{timestamp}.jpg"

        cv2.imwrite(os.path.join(output_dir, filename), processed_frame)
        cv2.imwrite(os.path.join(output_dir, f"mask_{filename}"), yolo_mask)
        cv2.imwrite(os.path.join(output_dir, f"floor_{filename}"), floor_mask)
        logger.info("finish")

    def capture_images(self, output_dir="/home/waryt/Desktop/image", max_images=None):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("capturing images to %s. Press 't' to save, 'q' to quit.", output_dir)

        count = 0
        try:
            while max_images is None or count < max_images:
                frame = self.picam2.capture_array()
                frame = self.fix_orientation(frame)
                cv2.imshow("Capture View", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("t"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(output_dir, f"image_{timestamp}_{count:04d}.jpg")
                    cv2.imwrite(filename, frame)
                    logger.info("saved %s", filename)
                    count += 1
                elif key == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("capture stopped by user")

    def capture_images_interval(self, output_dir="/home/waryt/Desktop/image", interval=1.0, max_images=None):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("capturing images to %s every %.1f second(s)", output_dir, interval)

        count = 0
        try:
            while max_images is None or count < max_images:
                start_time = time.monotonic()
                frame = self.picam2.capture_array()
                frame = self.fix_orientation(frame)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"image_{timestamp}_{count:04d}.jpg")
                cv2.imwrite(filename, frame)
                logger.info("saved %s", filename)

                count += 1
                sleep_time = interval - (time.monotonic() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("capture stopped by user")

    def close(self):
        if self.closed:
            return
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.closed = True
        logger.info("Camera and Windows closed.")


if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
