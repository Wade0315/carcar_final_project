import cv2
import numpy as np
import time
import logging
import os
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
    def __init__(self, width=320, height=240, camera_index=0, flip_code=-1):
        super().__init__(width, height, flip_code)
        self.camera_index = camera_index

        self.cap = cv2.VideoCapture(self.camera_index)
        self.closed = False

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        logger.info("camera activating...")
        time.sleep(2)

    def process_frame(self, frame):
        floor_mask, badminton_mask, candidate, target, find_ball, error = self.detect_frame(frame)
        self.visualize_frame(frame, floor_mask, candidate, target)
        return frame, floor_mask, badminton_mask, find_ball, error

    def draw_target(self, frame, target):
        target_cx = target["target_cx"]
        target_cy = target["target_cy"]

        cv2.circle(frame, (target_cx, target_cy), 6, (0, 255, 0), -1)
        cv2.putText(frame,"TARGET",(target_cx - 25, target_cy + 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
        cv2.line(frame,(self.width // 2, target_cy),(target_cx, target_cy),(0, 255, 0),2)
        cv2.putText(frame,f"error={target['error']}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
        logger.debug("target area=%s target_cx=%s target_cy=%s ratio=%.3f head_found=%s",target["area"],target_cx,target_cy,target["w_h_ratio"],target["head_found"])

    def visualize_frame(self, frame, floor_mask, candidate, target):
        self.draw_center_line(frame)
        self.draw_floor_boundary(frame, floor_mask)
        self.draw_candicate(frame, candidate)
        if target is not None:
            self.draw_target(frame, target)

    def draw_center_line(self, frame):
        cv2.line(frame,(self.width // 2, 0),(self.width // 2, self.height),(0, 255, 255),1)

    def draw_floor_boundary(self, frame, floor_mask):
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 255), 3)

    def draw_candicate(self, frame, candidate):
        for ball in candidate:
            box = cv2.boxPoints(ball["rect"])
            box = np.intp(box)

            target_cx = ball["target_cx"]
            target_cy = ball["target_cy"]
            rect_cx = ball["rect_cx"]
            rect_cy = ball["rect_cy"]

            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            cv2.circle(frame, (rect_cx, rect_cy), 3, (255, 255, 0), -1)
            if ball["head_found"]:
                cv2.circle(frame, (target_cx, target_cy), 4, (0, 0, 255), -1)
                cv2.putText(frame,"Head",(target_cx - 12, target_cy - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
            else:
                cv2.putText(frame,"Ball",(target_cx - 10, target_cy - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)

    def streaming(self):
        logger.info("Starting tracking... Press 'q' to quit.")

        at_frame = 0
        processed_frame = None
        floor_mask = None
        badminton_mask = None
        find_ball = False
        error = None

        try:
            while True:
                ret, raw_frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to read frame")
                    break

                raw_frame = cv2.resize(raw_frame, (self.width, self.height))
                raw_frame = self.fix_orientation(raw_frame)

                if at_frame % 3 == 0:
                    processed_frame, floor_mask, badminton_mask, find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error

                if processed_frame is not None:
                    cv2.imshow("Robot View", processed_frame)

                if badminton_mask is not None and floor_mask is not None:
                    display_mask = self.build_display_mask(floor_mask, badminton_mask)
                    cv2.imshow("White Mask", display_mask)

                at_frame += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def single_test(self, filename="test_capture.jpg"):
        logger.info("capturing photo...")

        ret, raw_frame = self.cap.read()

        if not ret:
            logger.warning("Failed to capture photo")
            return

        raw_frame = cv2.resize(raw_frame, (self.width, self.height))
        raw_frame = self.fix_orientation(raw_frame)

        processed_frame, floor_mask, badminton_mask, find_ball, error = self.process_frame(raw_frame)
        logger.info("find_ball=%s error=%s", find_ball, error)
        cv2.imwrite(filename, processed_frame)
        display_mask = self.build_display_mask(floor_mask, badminton_mask)
        cv2.imwrite(f"mask_{filename}", display_mask)

        logger.info("finish")

    def close(self):
        if self.closed:
            return
        self.cap.release()
        cv2.destroyAllWindows()
        self.closed = True
        logger.info("Camera and Windows closed.")

if __name__ == "__main__":
    setup_logging()
    with Camera(flip_code=None) as tracker:
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
