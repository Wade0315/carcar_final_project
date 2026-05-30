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
        yolo_mask = self.build_yolo_mask(frame, candidates)
        self.visualize_frame(frame, floor_mask, candidates, target)
        return frame, floor_mask, yolo_mask, find_ball, error

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
        logger.info("Starting YOLO debug view... Press 'q' to quit.")
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
                    #cv2.imshow("Floor Mask", floor_mask)
                    cv2.imshow("YOLO Mask", yolo_mask)

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
        cv2.imwrite(os.path.join(output_dir, f"yolo_mask_{filename}"), yolo_mask)
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

    def close(self):
        super().close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
