import logging
import os
import time

import cv2
import numpy as np

from camera_YOLO import Camera as YOLOCamera
from camera_YOLO import setup_logging

logger = logging.getLogger(__name__)


class Camera(YOLOCamera):
    def build_yolo_mask(self, frame, candidates):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["bbox"]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def process_frame(self, frame):
        floor_mask, candidates, target, find_ball, error = self.detect_frame(frame)
        logger.info("debug candidates=%s find_ball=%s error=%s", len(candidates), find_ball, error)
        yolo_mask = self.build_yolo_mask(frame, candidates)
        self.visualize_frame(frame, floor_mask, candidates, target, error)
        return frame, floor_mask, yolo_mask, find_ball, error, target

    def draw_target(self, frame, target):
        target_cx = target["target_cx"]
        target_cy = target["target_cy"]

        x1, y1, x2, y2 = target["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(frame, (target_cx, target_cy), 7, (0, 255, 0), -1)
        cv2.line(
            frame,
            (self.width // 2, target_cy),
            (target_cx, target_cy),
            (0, 255, 0),
            2,
        )

    def visualize_frame(self, frame, floor_mask, candidates, target, error):
        self.draw_center_line(frame)
        self.draw_candidates(frame, candidates)
        if target is not None:
            self.draw_target(frame, target)
        self.draw_error_text(frame, error)

    def draw_center_line(self, frame):
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (0, 255, 255), 1)

    def draw_candidates(self, frame, candidates):
        for ball in candidates:
            x1, y1, x2, y2 = ball["bbox"]
            target_cx = ball["target_cx"]
            target_cy = ball["target_cy"]
            color = (255, 128, 255) if ball.get("is_head") else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (target_cx, target_cy), 4, (0, 0, 255), -1)
            self.draw_candidate_label(frame, ball, color)

    def draw_candidate_label(self, frame, candidate, color):
        x1, y1, x2, y2 = candidate["bbox"]
        label = f"{candidate['class_name']} {candidate['confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.34
        thickness = 1
        text_w, text_h = cv2.getTextSize(label, font, scale, thickness)[0]

        label_x = x2 + 4
        label_y = y1 + text_h + 2
        if label_x + text_w >= self.width:
            label_x = max(0, x1 - text_w - 4)
        if label_y >= y2:
            label_y = max(text_h + 2, y1 - 4)
        if label_y - text_h < 0:
            label_y = min(self.height - 2, y2 + text_h + 4)

        cv2.putText(frame, label, (label_x, label_y), font, scale, color, thickness)

    def draw_error_text(self, frame, error):
        cv2.putText(
            frame,
            f"error={error}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if error is not None else (0, 255, 255),
            1,
        )

    def streaming(self):
        logger.info("Starting YOLO debug view... Press 'q' to quit.")
        last_frame_index = None
        processed_frame = None

        try:
            while True:
                raw_frame, capture_ms, frame_index = self.get_latest_frame(last_frame_index)
                last_frame_index = frame_index
                processing_started_at = time.perf_counter()
                floor_mask, candidates, target, find_ball, error = self.detect_frame(raw_frame)
                processing_ms = (time.perf_counter() - processing_started_at) * 1000
                self.record_performance(frame_index, capture_ms, processing_ms, find_ball, error)
                logger.info("debug candidates=%s find_ball=%s error=%s", len(candidates), find_ball, error)
                processed_frame = raw_frame.copy()
                try:
                    self.visualize_frame(processed_frame, floor_mask, candidates, target, error)
                except Exception:
                    logger.exception("failed to visualize frame")
                area = target["area"] if target is not None else None
                logger.info("find_ball=%s error=%s area=%s", find_ball, error, area)
                yield find_ball, error, target

                if processed_frame is not None:
                    cv2.imshow("Robot View", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def single_test(self, output_dir="/home/waryt/YOLO/image", filename=None):
        logger.info("capturing photo...")
        os.makedirs(output_dir, exist_ok=True)

        raw_frame, capture_ms, _ = self.get_latest_frame()
        detection_frame = raw_frame.copy()

        processing_started_at = time.perf_counter()
        processed_frame, floor_mask, yolo_mask, find_ball, error, target = self.process_frame(detection_frame)
        processing_ms = (time.perf_counter() - processing_started_at) * 1000
        area = target["area"] if target is not None else None
        logger.info("find_ball=%s error=%s area=%s", find_ball, error, area)

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"yolo_{timestamp}.jpg"

        photo_path = os.path.join(output_dir, filename)
        cv2.imwrite(photo_path, raw_frame)
        logger.info("saved photo %s", photo_path)

        save_debug = os.getenv("YOLO_SAVE_DEBUG_PHOTO", "").strip().lower() in {"1", "true", "yes"}
        if save_debug:
            debug_path = os.path.join(output_dir, f"debug_{filename}")
            cv2.imwrite(debug_path, processed_frame)
            logger.info("saved debug photo %s", debug_path)
        logger.info("finish")

    def log_camera_parameters(self):
        metadata = self.picam2.capture_metadata()
        logger.info(
            "camera params ExposureTime=%s us AnalogueGain=%s ColourGains=%s "
            "FrameDuration=%s us Lux=%s",
            metadata.get("ExposureTime"),
            metadata.get("AnalogueGain"),
            metadata.get("ColourGains"),
            metadata.get("FrameDuration"),
            metadata.get("Lux"),
        )

    def capture_images(self, output_dir="/home/waryt/YOLO/image", max_images=None):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("capturing images to %s. Press 't' to save, 'q' to quit.", output_dir)

        count = 0
        last_frame_index = None
        next_params_log_at = 0.0
        try:
            while max_images is None or count < max_images:
                frame, _, frame_index = self.get_latest_frame(last_frame_index)
                last_frame_index = frame_index
                now = time.monotonic()
                if now >= next_params_log_at:
                    self.log_camera_parameters()
                    next_params_log_at = now + 1.0
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

    def close(self):
        super().close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        tracker.single_test()
        for find_ball, error, target in tracker.streaming():
            if target is not None:
                logger.info("find_ball=%s error=%s area=%s", find_ball, error, target["area"])
            else:
                logger.info("find_ball=%s error=%s", find_ball, error)
