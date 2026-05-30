import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraBase:
    def __init__(self, width=320, height=240, flip_code=-1):
        self.width = width
        self.height = height
        self.flip_code = flip_code

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
                if self.lost_count <= self.max_lost_frames:
                    return True, self.last_error, None

                self.target_x = None
                self.target_y = None
                self.last_error = None
                return False, None, None

            self.target_x = target["target_cx"]
            self.target_y = target["target_cy"]
            error = target["error"]
            self.last_error = error
            self.lost_count = 0

            return True, error, target

        self.lost_count += 1
        if self.last_error is not None and self.lost_count <= self.max_lost_frames:
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
