import logging
import os
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
        self.camera_frame_duration_us = int(round(1_000_000 / self.camera_fps))
        self.lock_frame_duration = os.getenv("CAMERA_LOCK_FRAME_DURATION", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.frame_timeout_seconds = float(os.getenv("CAMERA_FRAME_TIMEOUT_SECONDS", "5"))
        self.max_frame_timeouts = max(1, int(os.getenv("CAMERA_MAX_FRAME_TIMEOUTS", "3")))
        self.frame_timeout_count = 0

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
        self.max_lost_frames = 2
        self.max_tracking_distance = 50
        self.target_area_dominance_ratio = max(
            1.0,
            float(os.getenv("YOLO_TARGET_AREA_DOMINANCE_RATIO", "1.4")),
        )
        self.head_only_min_area_ratio = max(
            0.0,
            float(os.getenv("YOLO_HEAD_ONLY_MIN_AREA_RATIO", "0.01")),
        )
        default_head_only_min_area = int(
            self.width * self.height * self.head_only_min_area_ratio
        )
        self.head_only_min_area = max(
            1,
            int(os.getenv("YOLO_HEAD_ONLY_MIN_AREA", str(default_head_only_min_area))),
        )
        self.head_end_band_ratio = 0.25
        self.head_width_ratio = 0.85
        self.picam2 = None
        self.closed = False
        self.latest_frame = None
        self.latest_capture_ms = None
        self.latest_capture_gap_ms = None
        self.latest_capture_completed_at = None
        self.latest_frame_index = -1
        self.current_capture_gap_ms = None
        self.current_frame_age_ms = None
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_ready = threading.Event()
        self.capture_stop = threading.Event()
        self.capture_error = None
        self.capture_thread = None
        logger.info(
            "camera base config width=%s height=%s flip_code=%s frame_interval=%s "
            "camera_fps=%s exposure_time_us=%s frame_duration_us=%s lock_frame_duration=%s "
            "frame_timeout_seconds=%s max_frame_timeouts=%s area_dominance_ratio=%.2f "
            "head_only_min_area=%s head_only_min_area_ratio=%.4f",
            self.width,
            self.height,
            self.flip_code,
            self.frame_interval,
            self.camera_fps,
            self.exposure_time_us,
            self.camera_frame_duration_us,
            self.lock_frame_duration,
            self.frame_timeout_seconds,
            self.max_frame_timeouts,
            self.target_area_dominance_ratio,
            self.head_only_min_area,
            self.head_only_min_area_ratio,
        )

    def open_camera(self, warmup_seconds=1, buffer_count=3, lock_controls=True):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is required for camera capture. Use cameraFAKE.py for local image preview."
            ) from exc

        self.picam2 = Picamera2()
        camera_controls = {"FrameRate": self.camera_fps}
        if self.lock_frame_duration:
            camera_controls["FrameDurationLimits"] = (
                self.camera_frame_duration_us,
                self.camera_frame_duration_us,
            )

        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)},
            controls=camera_controls,
            buffer_count=buffer_count,
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.closed = False

        logger.info(
            "camera activating size=%sx%s fps=%s frame_duration_us=%s buffer_count=%s "
            "warmup_seconds=%s",
            self.width,
            self.height,
            self.camera_fps,
            self.camera_frame_duration_us if self.lock_frame_duration else None,
            buffer_count,
            warmup_seconds,
        )
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
        if self.lock_frame_duration:
            controls["FrameDurationLimits"] = (
                self.camera_frame_duration_us,
                self.camera_frame_duration_us,
            )
        if self.exposure_time_us is not None:
            controls["ExposureTime"] = self.exposure_time_us
        if analogue_gain is not None:
            controls["AnalogueGain"] = analogue_gain
        if colour_gains is not None:
            controls["ColourGains"] = colour_gains

        self.picam2.set_controls(controls)
        logger.info(
            "lock camera ExposureTime=%s us (measured=%s us) FrameDurationLimits=%s "
            "AnalogueGain=%s (measured=%s) ColourGains=%s",
            self.exposure_time_us,
            measured_exposure_time,
            controls.get("FrameDurationLimits"),
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
        last_capture_completed_at = None
        try:
            while not self.capture_stop.is_set():
                capture_started_at = time.perf_counter()
                frame = self.picam2.capture_array()
                frame = self.fix_orientation(frame)
                capture_completed_at = time.perf_counter()
                capture_ms = (capture_completed_at - capture_started_at) * 1000
                capture_gap_ms = None
                if last_capture_completed_at is not None:
                    capture_gap_ms = (capture_completed_at - last_capture_completed_at) * 1000
                last_capture_completed_at = capture_completed_at
                with self.latest_frame_lock:
                    self.latest_frame = frame
                    self.latest_capture_ms = capture_ms
                    self.latest_capture_gap_ms = capture_gap_ms
                    self.latest_capture_completed_at = capture_completed_at
                    self.latest_frame_index = frame_index
                self.latest_frame_ready.set()
                if frame_index == 0:
                    logger.info("first camera frame captured capture_ms=%.1f", capture_ms)
                elif capture_ms > self.camera_frame_period_ms * 2:
                    logger.debug(
                        "slow camera capture frame=%s capture_ms=%.1f capture_gap_ms=%s "
                        "expected_period_ms=%.1f",
                        frame_index,
                        capture_ms,
                        "%.1f" % capture_gap_ms if capture_gap_ms is not None else None,
                        self.camera_frame_period_ms,
                    )
                frame_index += 1
        except Exception as exc:
            self.capture_error = exc
            self.latest_frame_ready.set()
            if not self.capture_stop.is_set():
                logger.exception("camera capture thread failed")

    def get_latest_frame(self, after_frame_index=None, timeout=None):
        if timeout is None:
            timeout = self.frame_timeout_seconds
        while True:
            if self.capture_error is not None:
                raise RuntimeError("camera capture thread failed") from self.capture_error
            with self.latest_frame_lock:
                frame = self.latest_frame
                capture_ms = self.latest_capture_ms
                capture_gap_ms = self.latest_capture_gap_ms
                capture_completed_at = self.latest_capture_completed_at
                frame_index = self.latest_frame_index
                if frame is not None and (after_frame_index is None or frame_index > after_frame_index):
                    self.frame_timeout_count = 0
                    retrieved_at = time.perf_counter()
                    self.current_capture_gap_ms = capture_gap_ms
                    self.current_frame_age_ms = (
                        (retrieved_at - capture_completed_at) * 1000
                        if capture_completed_at is not None
                        else None
                    )
                    return frame.copy(), capture_ms, frame_index
                self.latest_frame_ready.clear()
            if self.capture_stop.is_set():
                raise RuntimeError("camera capture thread stopped")
            if not self.latest_frame_ready.wait(timeout):
                self.frame_timeout_count += 1
                capture_thread_alive = (
                    self.capture_thread is not None and self.capture_thread.is_alive()
                )
                logger.warning(
                    "waiting for camera frame timed out after %.1fs count=%s/%s "
                    "after_frame_index=%s latest_frame_index=%s capture_thread_alive=%s",
                    timeout,
                    self.frame_timeout_count,
                    self.max_frame_timeouts,
                    after_frame_index,
                    frame_index,
                    capture_thread_alive,
                )
                if self.frame_timeout_count < self.max_frame_timeouts:
                    continue
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
        candidates = candidate or []
        target = None
        if self.has_locked_target():
            tracking_candidates = self.select_locked_tracking_candidates(candidates)
            if tracking_candidates:
                target = self.choose_locked_tracking_candidate(tracking_candidates)
        else:
            tracking_candidates = self.select_tracking_candidates(candidates)
            if tracking_candidates:
                target = self.choose_tracking_candidate(tracking_candidates)

        if target is not None:
            distance = self.previous_target_distance(target)

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
                    "target jumped %.1f px; reset lock and reacquire after lost_count=%s",
                    distance,
                    self.lost_count,
                )
                self.target_x = None
                self.target_y = None
                tracking_candidates = self.select_tracking_candidates(candidates)
                if not tracking_candidates:
                    self.last_error = None
                    return False, None, None
                target = self.choose_tracking_candidate(tracking_candidates)

            self.target_x = target["target_cx"]
            self.target_y = target["target_cy"]
            error = target["error"]
            self.last_error = error
            self.lost_count = 0
            logger.debug(
                "target selected tier=%s source=%s class=%s confidence=%.3f area=%s bbox=%s error=%s",
                self.candidate_tracking_tier(target),
                target.get("source"),
                target.get("class_name"),
                target.get("confidence", 0),
                self.candidate_selection_area(target),
                target.get("bbox"),
                error,
            )

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

    def has_locked_target(self):
        return self.target_x is not None and self.target_y is not None

    def select_tracking_candidates(self, candidates):
        grouped_candidates = [
            candidate for candidate in candidates
            if self.is_grouped_candidate(candidate)
        ]
        if grouped_candidates:
            return grouped_candidates

        head_candidates = [
            candidate for candidate in candidates
            if self.is_qualified_head_only_candidate(candidate)
        ]
        if head_candidates:
            return head_candidates

        body_candidates = [
            candidate for candidate in candidates
            if not candidate.get("is_head")
        ]
        if body_candidates:
            return body_candidates

        if candidates:
            logger.debug(
                "discard candidates: no grouped/body candidates and no head-only area >= %s",
                self.head_only_min_area,
            )
        return []

    def select_locked_tracking_candidates(self, candidates):
        tracking_candidates = [
            candidate for candidate in candidates
            if (
                self.is_grouped_candidate(candidate)
                or not candidate.get("is_head")
                or self.is_qualified_head_only_candidate(candidate)
            )
        ]
        if tracking_candidates:
            return tracking_candidates

        if candidates:
            logger.debug(
                "discard locked candidates: only head-only candidates below area %s",
                self.head_only_min_area,
            )
        return []

    def choose_locked_tracking_candidate(self, candidates):
        target = min(
            candidates,
            key=lambda candidate: (
                self.previous_target_distance_sq(candidate),
                -self.candidate_selection_area(candidate),
            ),
        )
        logger.debug(
            "target selected by locked distance tier=%s area=%s distance_sq=%.1f",
            self.candidate_tracking_tier(target),
            self.candidate_selection_area(target),
            self.previous_target_distance_sq(target),
        )
        return self.mark_candidate_selection(target, "locked_distance")

    def choose_tracking_candidate(self, candidates):
        if len(candidates) == 1:
            return self.mark_candidate_selection(candidates[0], "single_candidate")

        by_area = sorted(
            candidates,
            key=lambda candidate: self.candidate_selection_area(candidate),
            reverse=True,
        )
        largest = by_area[0]
        second_largest = by_area[1]
        largest_area = self.candidate_selection_area(largest)
        second_largest_area = max(1, self.candidate_selection_area(second_largest))

        if largest_area >= second_largest_area * self.target_area_dominance_ratio:
            logger.debug(
                "target selected by area dominance tier=%s largest_area=%s second_area=%s ratio=%.2f",
                self.candidate_tracking_tier(largest),
                largest_area,
                second_largest_area,
                largest_area / second_largest_area,
            )
            return self.mark_candidate_selection(largest, "area_dominance")

        target = min(
            candidates,
            key=lambda candidate: self.candidate_reference_distance_sq(candidate),
        )
        logger.debug(
            "target selected by reference distance tier=%s area=%s distance_sq=%.1f",
            self.candidate_tracking_tier(target),
            self.candidate_selection_area(target),
            self.candidate_reference_distance_sq(target),
        )
        return self.mark_candidate_selection(target, "reference_distance")

    def is_grouped_candidate(self, candidate):
        return (
            candidate.get("is_head")
            and candidate.get("grouped_area") is not None
            and candidate.get("ball_area") is not None
        )

    def is_qualified_head_only_candidate(self, candidate):
        return (
            candidate.get("is_head")
            and not self.is_grouped_candidate(candidate)
            and self.candidate_selection_area(candidate) >= self.head_only_min_area
        )

    def candidate_tracking_tier(self, candidate):
        if self.is_grouped_candidate(candidate):
            return "grouped"
        if candidate.get("is_head"):
            return "head_only"
        return "body_only"

    def candidate_selection_area(self, candidate):
        if self.is_grouped_candidate(candidate):
            return candidate.get("grouped_area") or candidate.get("area") or 0
        if candidate.get("is_head"):
            return candidate.get("area") or 0
        return candidate.get("ball_area") or candidate.get("area") or 0

    def candidate_reference_distance_sq(self, candidate):
        if self.target_x is not None and self.target_y is not None:
            reference_x = self.target_x
            reference_y = self.target_y
            point_x = candidate["target_cx"]
            point_y = candidate["target_cy"]
        else:
            reference_x = self.width / 2
            reference_y = self.height / 2
            point_x, point_y = self.candidate_center_point(candidate)

        return (point_x - reference_x) ** 2 + (point_y - reference_y) ** 2

    def candidate_center_point(self, candidate):
        if self.is_grouped_candidate(candidate):
            return (
                candidate.get("grouped_cx") or candidate["target_cx"],
                candidate.get("grouped_cy") or candidate["target_cy"],
            )
        return candidate["target_cx"], candidate["target_cy"]

    def previous_target_distance(self, candidate):
        if self.target_x is None or self.target_y is None:
            return 0
        return self.previous_target_distance_sq(candidate) ** 0.5

    def previous_target_distance_sq(self, candidate):
        if self.target_x is None or self.target_y is None:
            return 0
        return (
            (candidate["target_cx"] - self.target_x) ** 2
            + (candidate["target_cy"] - self.target_y) ** 2
        )

    def mark_candidate_selection(self, candidate, mode):
        candidate["tracking_tier"] = self.candidate_tracking_tier(candidate)
        candidate["selection_mode"] = mode
        candidate["selection_area"] = self.candidate_selection_area(candidate)
        if self.has_locked_target():
            candidate["selection_distance_sq"] = self.previous_target_distance_sq(candidate)
        else:
            candidate["selection_distance_sq"] = self.candidate_reference_distance_sq(candidate)
        return candidate

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
        logger.info("Camera closed last_frame_index=%s", self.latest_frame_index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
