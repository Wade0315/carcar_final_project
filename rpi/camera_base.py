import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraBase:
    def __init__(self, width=320, height=240, flip_code=-1):
        self.width = width
        self.height = height
        self.flip_code = flip_code

        self.lower_white = np.array([0, 180, 0])
        self.upper_white = np.array([180, 255, 40])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)

        self.lower_floor = np.array([35, 40, 20])
        self.upper_floor = np.array([95, 255, 180])
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

    def estimate_head_from_long_axis(self, cnt, rect):
        points = cnt.reshape(-1, 2).astype(np.float32)
        if len(points) < 6:
            return None

        center = np.array(rect[0], dtype=np.float32)
        box = cv2.boxPoints(rect).astype(np.float32)
        edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
        long_axis = max(edges, key=lambda edge: np.linalg.norm(edge))
        axis_length = np.linalg.norm(long_axis)
        if axis_length <= 0:
            return None

        long_axis = long_axis / axis_length
        short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float32)

        relative_points = points - center
        long_projection = relative_points @ long_axis
        short_projection = relative_points @ short_axis
        min_projection = float(np.min(long_projection))
        max_projection = float(np.max(long_projection))
        projection_span = max_projection - min_projection
        if projection_span < 8:
            return None

        band_width = max(projection_span * self.head_end_band_ratio, 4)
        low_end = short_projection[long_projection <= min_projection + band_width]
        high_end = short_projection[long_projection >= max_projection - band_width]
        if len(low_end) < 3 or len(high_end) < 3:
            return None

        low_width = float(np.max(low_end) - np.min(low_end))
        high_width = float(np.max(high_end) - np.min(high_end))
        narrow_width = min(low_width, high_width)
        wide_width = max(low_width, high_width)
        if wide_width <= 0 or narrow_width / wide_width > self.head_width_ratio:
            return None

        head_projection = min_projection if low_width < high_width else max_projection
        head_point = center + long_axis * head_projection
        head_x = int(round(head_point[0]))
        head_y = int(round(head_point[1]))
        if not (0 <= head_x < self.width and 0 <= head_y < self.height):
            return None

        return head_x, head_y

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

    def build_badminton_mask(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        badminton_mask = cv2.inRange(hls, self.lower_white, self.upper_white)
        badminton_mask = cv2.morphologyEx(badminton_mask, cv2.MORPH_OPEN, self.kernel_open)
        badminton_mask = cv2.morphologyEx(badminton_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return badminton_mask

    def build_display_mask(self, floor_mask, badminton_mask):
            if not self.has_floor(floor_mask):
                return badminton_mask
            return cv2.bitwise_and(badminton_mask, badminton_mask, mask=floor_mask)

    def has_floor(self, floor_mask):
        return cv2.countNonZero(floor_mask) >= self.min_floor_area
    def detect_frame(self, frame):
        candidate = []
        floor_mask = self.build_floor_mask(frame)
        badminton_mask = self.build_badminton_mask(frame)
        contours, _ = cv2.findContours(badminton_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            self.contour_dealing(cnt, candidate, floor_mask)

        find_ball, error, target = self.choose_ball(candidate)
        return floor_mask, badminton_mask, candidate, target, find_ball, error

    def process_frame(self, frame):
        floor_mask, badminton_mask, _, _, find_ball, error = self.detect_frame(frame)
        return frame, floor_mask, badminton_mask, find_ball, error

    def contour_dealing(self, cnt, candidate, floor_mask):
        area = cv2.contourArea(cnt)
        if not (800 < area < 10000):
            return

        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            return

        w_h_ratio = min(w, h) / max(w, h)
        if w_h_ratio <= 0.35:
            return

        rect_cx, rect_cy = int(rect[0][0]), int(rect[0][1])
        if not (0 <= rect_cx < self.width and 0 <= rect_cy < self.height):
            return
        if self.has_floor(floor_mask) and floor_mask[rect_cy, rect_cx] == 0:
            return

        head = self.estimate_head_from_long_axis(cnt, rect)
        head_found = head is not None
        if head_found:
            target_cx, target_cy = head
        else:
            target_cx, target_cy = rect_cx, rect_cy

        error_from_center = target_cx - self.width // 2
        candidate.append({
            "contour": cnt,
            "rect": rect,
            "area": area,
            "rect_cx": rect_cx,
            "rect_cy": rect_cy,
            "target_cx": target_cx,
            "target_cy": target_cy,
            "error": error_from_center,
            "w_h_ratio": w_h_ratio,
            "head_found": head_found
        })

    def choose_ball(self, candidate):
        if len(candidate) > 0:
            if self.target_x is None:
                target = min(candidate, key=lambda b: abs(b["error"]))
                distance = 0
            else:
                target = min(
                    candidate,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
