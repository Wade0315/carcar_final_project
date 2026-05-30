import cv2
import numpy as np
import time
import logging
import os
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
    def __init__(self, width=320, height=240, flip_code=-1):
        super().__init__(width, height, flip_code)

        self.picam2 = Picamera2()
        self.closed = False
        self.mask_modes = ("white", "floor", "ring")
        self.mask_mode_index = 0
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)},
            controls = {"FrameRate": 30},
            buffer_count = 3
        )
        self.picam2.configure(config)
        self.picam2.start()
        logger.info("camera activating...")
        time.sleep(2)

        metadata = self.picam2.capture_metadata()
        exposure_time = metadata.get("ExposureTime")
        analogue_gain = metadata.get("AnalogueGain")
        colour_gains = metadata.get("ColourGains")

        logger.info(
            "lock camera ExposureTime=%s AnalogueGain=%s ColourGains=%s",
            exposure_time,
            analogue_gain,
            colour_gains,
        )

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

    def process_frame(self, frame):
        floor_mask, badminton_mask, candidate, target, find_ball, error = self.detect_frame(frame)
        ring_mask = self.build_black_ring_mask(frame, floor_mask)
        self.visualize_frame(frame, floor_mask, candidate, target)
        return frame, floor_mask, badminton_mask, ring_mask, find_ball, error

    def current_mask_mode(self):
        return self.mask_modes[self.mask_mode_index]

    def switch_mask_mode(self):
        self.mask_mode_index = (self.mask_mode_index + 1) % len(self.mask_modes)
        logger.info("mask mode=%s", self.current_mask_mode())

    def select_display_mask(self, floor_mask, badminton_mask, ring_mask):
        mode = self.current_mask_mode()
        if mode == "white":
            return "White Mask", badminton_mask
        if mode == "floor":
            return "Floor Mask", floor_mask
        return "Ring Mask", ring_mask

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
        logger.debug(
            "target area=%s target_cx=%s target_cy=%s ratio=%.3f head_found=%s",
            target["area"],
            target_cx,
            target_cy,
            target["w_h_ratio"],
            target["head_found"],
        )

    def visualize_frame(self, frame, floor_mask, candidate, target):
        self.draw_center_line(frame)
        self.draw_floor_boundary(frame, floor_mask)
        self.draw_candidate(frame, candidate)
        if target is not None:
            self.draw_target(frame, target)

    def draw_center_line(self, frame):
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (0, 255, 255), 1)

    def draw_floor_boundary(self, frame, floor_mask):
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 255), 3)

    def draw_candidate(self, frame, candidate):
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
                cv2.putText(
                    frame,
                    "Head",
                    (target_cx - 12, target_cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            else:
                cv2.putText(
                    frame,
                    "Ball",
                    (target_cx - 10, target_cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

    def streaming(self):
        logger.info("Starting tracking... Press 'q' to quit.")
        at_frame = 0
        processed_frame = None
        floor_mask = None
        badminton_mask = None
        ring_mask = None

        try:
            while True:
                raw_frame = self.picam2.capture_array()
                raw_frame = self.fix_orientation(raw_frame)

                if at_frame % 3 == 0:
                    processed_frame, floor_mask, badminton_mask, ring_mask, find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error

                if processed_frame is not None:
                    cv2.imshow("Robot View", processed_frame)

                if floor_mask is not None and badminton_mask is not None and ring_mask is not None:
                    window_name, display_mask = self.select_display_mask(floor_mask, badminton_mask, ring_mask)
                    cv2.imshow("Mask View", display_mask)
                    cv2.setWindowTitle("Mask View", window_name)

                at_frame += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("m"):
                    self.switch_mask_mode()
                elif key == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def single_test(self, output_dir="/home/waryt/Desktop/image", filename="test_capture.jpg"):
        logger.info("capturing photo...")
        raw_frame = self.picam2.capture_array()
        raw_frame = self.fix_orientation(raw_frame)

        processed_frame, floor_mask, badminton_mask, ring_mask, find_ball, error = self.process_frame(raw_frame)
        logger.info("find_ball=%s error=%s", find_ball, error)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"image_{timestamp}.jpg")
        cv2.imwrite(filename, raw_frame)
        logger.info("saved %s", filename)
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
        #tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
