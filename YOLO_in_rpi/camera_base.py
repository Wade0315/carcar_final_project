import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraBase:
    def __init__(
        self,
        width=320,
        height=240,
        flip_code=-1,
        frame_interval=1,
        camera_fps=30,
        exposure_time_us=None,
    ):
        self.width = width
        self.height = height
        self.flip_code = flip_code
        self.frame_interval = frame_interval
        self.camera_fps = camera_fps
        self.exposure_time_us = exposure_time_us
        self.camera_frame_period_ms = self.frame_interval / self.camera_fps * 1000

        self.lower_floor = np.array([35, 90, 35])
        self.upper_floor = np.array([95, 255, 185])
        self.floor_kernel_open = np.ones((5, 5), np.uint8)
        self.floor_kernel_close = np.ones((21, 21), np.uint8)
        self.floor_boundary_margin = 0
        self.floor_bottom_band_ratio = 0.75
        self.min_floor_area = int(self.width * self.height * 0.03)

        self.target_x = None
        self.target_y = None
        self.last_error = None
        self.lost_count = 0
        self.max_lost_frames = 5
        self.max_tracking_distance = 50
        self.head_end_band_ratio = 0.25
        self.head_width_ratio = 0.85
        self.picam2 = None
        self.closed = False
        self.latest_frame = None
        self.latest_capture_ms = None
        self.latest_frame_index = -1
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_ready = threading.Event()
        self.capture_stop = threading.Event()
        self.capture_error = None
        self.capture_thread = None

    def open_camera(self, warmup_seconds=1, buffer_count=3, lock_controls=True):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is required for camera capture. Use cameraFAKE.py for local image preview."
            ) from exc

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)},
            controls={"FrameRate": self.camera_fps},
            buffer_count=buffer_count,
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.closed = False

        logger.info("camera activating...")
        if warmup_seconds > 0:
            time.sleep(warmup_seconds)
        if lock_controls:
            self.lock_current_camera_controls()

    def lock_current_camera_controls(self):
        if self.picam2 is None:
            raise RuntimeError("camera is not open")

        for _ in range(10):
            self.picam2.capture_array()
        metadata = self.picam2.capture_metadata()
        measured_exposure_time = metadata.get("ExposureTime")
        measured_analogue_gain = metadata.get("AnalogueGain")
        analogue_gain = measured_analogue_gain + 1 if measured_analogue_gain is not None else None
        colour_gains = metadata.get("ColourGains")

        controls = {
            "AeEnable": False,
            "AwbEnable": False,
        }
        if self.exposure_time_us is not None:
            controls["ExposureTime"] = self.exposure_time_us
        if analogue_gain is not None:
            controls["AnalogueGain"] = analogue_gain
        if colour_gains is not None:
            controls["ColourGains"] = colour_gains

        self.picam2.set_controls(controls)
        logger.info(
            "lock camera ExposureTime=%s us (measured=%s us) AnalogueGain=%s (measured=%s) ColourGains=%s",
            self.exposure_time_us,
            measured_exposure_time,
            analogue_gain,
            measured_analogue_gain,
            colour_gains,
        )

    def start_frame_capture(self):
        if self.picam2 is None:
            raise RuntimeError("camera is not open")
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return

        self.capture_stop.clear()
        self.capture_error = None
        self.capture_thread = threading.Thread(
            target=self.capture_latest_frames,
            name="camera-capture",
            daemon=True,
        )
        self.capture_thread.start()
        logger.info("camera capture thread started")

    def capture_latest_frames(self):
        frame_index = 0
        try:
            while not self.capture_stop.is_set():
                capture_started_at = time.perf_counter()
                frame = self.picam2.capture_array()
                frame = self.fix_orientation(frame)
                capture_ms = (time.perf_counter() - capture_started_at) * 1000
                with self.latest_frame_lock:
                    self.latest_frame = frame
                    self.latest_capture_ms = capture_ms
                    self.latest_frame_index = frame_index
                self.latest_frame_ready.set()
                frame_index += 1
        except Exception as exc:
            self.capture_error = exc
            self.latest_frame_ready.set()
            if not self.capture_stop.is_set():
                logger.exception("camera capture thread failed")

    def get_latest_frame(self, after_frame_index=None, timeout=2):
        while True:
            if self.capture_error is not None:
                raise RuntimeError("camera capture thread failed") from self.capture_error
            with self.latest_frame_lock:
                frame = self.latest_frame
                capture_ms = self.latest_capture_ms
                frame_index = self.latest_frame_index
                if frame is not None and (after_frame_index is None or frame_index > after_frame_index):
                    return frame, capture_ms, frame_index
                self.latest_frame_ready.clear()
            if self.capture_stop.is_set():
                raise RuntimeError("camera capture thread stopped")
            if not self.latest_frame_ready.wait(timeout):
                raise TimeoutError("timed out waiting for camera frame")

    def reset_tracking(self):
        self.target_x = None
        self.target_y = None
        self.last_error = None
        self.lost_count = 0

    def fix_orientation(self, frame):
        if self.flip_code is None:
            return frame
        return cv2.flip(frame, self.flip_code)

    def build_floor_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_floor, self.upper_floor)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.floor_kernel_open)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.floor_kernel_close)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        floor_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if not contours:
            return floor_mask

        bottom_y = int(self.height * self.floor_bottom_band_ratio)
        floor_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            _, y, _, h = cv2.boundingRect(cnt)
            reaches_bottom_band = y + h >= bottom_y

            if area >= self.min_floor_area and reaches_bottom_band:
                floor_contours.append(cnt)

        if not floor_contours:
            return floor_mask

        floor_hulls = [cv2.convexHull(cnt) for cnt in floor_contours]
        cv2.drawContours(floor_mask, floor_hulls, -1, 255, -1)
        if self.floor_boundary_margin > 0:
            margin_kernel = np.ones(
                (self.floor_boundary_margin, self.floor_boundary_margin),
                np.uint8
            )
            floor_mask = cv2.erode(floor_mask, margin_kernel)
        return floor_mask

    def choose_ball(self, candidate):
        if len(candidate) > 0:
            tracking_candidates = self.select_tracking_candidates(candidate)

            if self.target_x is None:
                target = min(tracking_candidates, key=lambda b: abs(b["error"]))
                distance = 0
            else:
                target = min(
                    tracking_candidates,
                    key=lambda b: (b["target_cx"] - self.target_x) ** 2
                    + (b["target_cy"] - self.target_y) ** 2
                )
                distance = (
                    (target["target_cx"] - self.target_x) ** 2
                    + (target["target_cy"] - self.target_y) ** 2
                ) ** 0.5

            if self.target_x is not None and distance > self.max_tracking_distance:
                self.lost_count += 1
                if self.last_error is not None and self.lost_count <= self.max_lost_frames:
                    logger.debug(
                        "target jumped %.1f px; reuse last_error=%s lost_count=%s/%s",
                        distance,
                        self.last_error,
                        self.lost_count,
                        self.max_lost_frames,
                    )
                    return True, self.last_error, None

                logger.debug(
                    "target jumped %.1f px; reacquire candidate after lost_count=%s",
                    distance,
                    self.lost_count,
                )

            self.target_x = target["target_cx"]
            self.target_y = target["target_cy"]
            error = target["error"]
            self.last_error = error
            self.lost_count = 0

            return True, error, target

        self.lost_count += 1
        if self.last_error is not None and self.lost_count <= self.max_lost_frames:
            logger.debug(
                "target missing; reuse last_error=%s lost_count=%s/%s",
                self.last_error,
                self.lost_count,
                self.max_lost_frames,
            )
            return True, self.last_error, None

        self.target_x = None
        self.target_y = None
        self.last_error = None
        return False, None, None

    def select_tracking_candidates(self, candidates):
        head_candidates = [candidate for candidate in candidates if candidate.get("is_head")]
        if head_candidates:
            return head_candidates
        return candidates

    def close(self):
        if self.closed:
            return

        self.capture_stop.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1)
        if self.picam2 is not None:
            self.picam2.stop()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1)
        self.closed = True
        logger.info("Camera closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
