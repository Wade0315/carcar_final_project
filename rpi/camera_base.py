import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraBase:
    def __init__(self, width=320, height=240, flip_code=-1):
        self.width = width
        self.height = height
        self.flip_code = flip_code

        self.lower_white = np.array([0, 166, 0])
        self.upper_white = np.array([180, 255, 174])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)

        self.lower_floor = np.array([35, 90, 20])
        self.upper_floor = np.array([95, 255, 190])
        self.floor_kernel_open = np.ones((5, 5), np.uint8)
        self.floor_kernel_close = np.ones((21, 21), np.uint8)
        self.floor_boundary_margin = 0
        self.floor_bottom_band_ratio = 0.75
        self.min_floor_area = int(self.width * self.height * 0.03)

        self.ring_v_max = 80
        self.ring_kernel_open = np.ones((1, 1), np.uint8)
        self.ring_kernel_close = np.ones((3, 3), np.uint8)
        self.ring_floor_top_min_buffer = 8
        self.ring_floor_top_max_buffer = 40
        self.ring_above_floor_top = 15
        self.ring_below_floor_top = 55
        self.ring_min_area = 20
        self.ring_max_area = 1200
        self.ring_max_size = 70
        self.ring_white_roi_x = 70
        self.ring_white_roi_up = 45
        self.ring_white_roi_down = 35
        self.ring_white_min_area = 120
        self.ring_white_min_nonline_area = 100
        self.ring_max_width = 50
        self.ring_max_height = 50
        self.ring_white_near_radius = 8
        self.ring_floor_touch_x = 5
        self.ring_floor_touch_down = 12

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

    def ring_floor_top_buffer(self, floor_top_y):
        floor_height = self.height - floor_top_y
        floor_ratio = floor_height / self.height
        buffer_range = self.ring_floor_top_max_buffer - self.ring_floor_top_min_buffer
        buffer = self.ring_floor_top_max_buffer - int(buffer_range * floor_ratio)
        return max(self.ring_floor_top_min_buffer, min(self.ring_floor_top_max_buffer, buffer))

    def build_floor_top_search_mask(self, floor_mask):
        if cv2.countNonZero(floor_mask) == 0:
            return np.full((self.height, self.width), 255, dtype=np.uint8)

        search_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for x in range(self.width):
            floor_y = np.flatnonzero(floor_mask[:, x])
            if len(floor_y) == 0:
                continue
            floor_top_y = int(floor_y[0])
            top_y = max(floor_top_y - self.ring_floor_top_buffer(floor_top_y), 0)
            bottom_y = min(floor_top_y + self.ring_below_floor_top, self.height)
            search_mask[top_y:bottom_y, x] = 255
        return search_mask

    def build_black_ring_mask(self, frame, floor_mask):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ring_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([180, 255, self.ring_v_max])
        )
        search_mask = self.build_floor_top_search_mask(floor_mask)
        ring_mask = cv2.bitwise_and(ring_mask, ring_mask, mask=search_mask)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, self.ring_kernel_open)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, self.ring_kernel_close)
        return ring_mask

    def is_near_floor_top(self, floor_mask, x, y):
        if cv2.countNonZero(floor_mask) == 0:
            return True
        if not (0 <= x < self.width):
            return False

        floor_y = np.flatnonzero(floor_mask[:, x])
        if len(floor_y) == 0:
            return False

        top_y = int(floor_y[0])
        return top_y - self.ring_above_floor_top <= y <= top_y + self.ring_below_floor_top

    def find_black_ring_candidates(self, ring_mask, floor_mask):
        contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rings = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.ring_min_area <= area <= self.ring_max_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.ring_max_size or h > self.ring_max_size:
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw == 0 or rh == 0:
                continue

            ratio = min(rw, rh) / max(rw, rh)
            if ratio < 0.15:
                continue

            center_x = int(rect[0][0])
            center_y = int(rect[0][1])
            if not self.is_near_floor_top(floor_mask, center_x, center_y):
                continue

            rings.append({
                "contour": cnt,
                "rect": rect,
                "area": area,
                "bbox": (x, y, w, h),
                "center_x": center_x,
                "center_y": center_y,
                "ratio": ratio,
            })

        rings.sort(key=lambda item: item["area"], reverse=True)
        return rings

    def crop_ring_white_roi(self, badminton_mask, ring):
        cx = ring["center_x"]
        cy = ring["center_y"]
        x1 = max(cx - self.ring_white_roi_x, 0)
        x2 = min(cx + self.ring_white_roi_x, self.width)
        y1 = max(cy - self.ring_white_roi_up, 0)
        y2 = min(cy + self.ring_white_roi_down, self.height)
        return badminton_mask[y1:y2, x1:x2]

    def horizontal_line_stats(self, mask):
        white_area = cv2.countNonZero(mask)
        if white_area == 0:
            return 0, 0

        kernel_width = max(mask.shape[1] // 3, 15)
        horizontal_kernel = np.ones((1, kernel_width), np.uint8)
        horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
        line_area = cv2.countNonZero(horizontal_lines)
        return line_area / white_area, white_area - line_area

    def ring_bottom_in_floor(self, ring, floor_mask):
        if cv2.countNonZero(floor_mask) == 0:
            return True

        x, y, w, h = ring["bbox"]
        bottom_x = x + w // 2
        bottom_y = y + h - 1
        if not (0 <= bottom_x < self.width and 0 <= bottom_y < self.height):
            return False

        x1 = max(bottom_x - self.ring_floor_touch_x, 0)
        x2 = min(bottom_x + self.ring_floor_touch_x + 1, self.width)
        y1 = bottom_y
        y2 = min(bottom_y + self.ring_floor_touch_down + 1, self.height)
        return cv2.countNonZero(floor_mask[y1:y2, x1:x2]) > 0

    def validate_black_ring(self, ring, badminton_mask, floor_mask):
        roi = self.crop_ring_white_roi(badminton_mask, ring)
        white_area = cv2.countNonZero(roi)
        _, nonline_area = self.horizontal_line_stats(roi)

        x, y, w, h = ring["bbox"]
        size_ok = w <= self.ring_max_width and h <= self.ring_max_height
        near_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.ring_white_near_radius * 2 + 1, self.ring_white_near_radius * 2 + 1)
        )
        near_white = cv2.dilate(badminton_mask, near_kernel)
        touches_white = cv2.countNonZero(near_white[y:y + h, x:x + w]) > 0
        bottom_in_floor = self.ring_bottom_in_floor(ring, floor_mask)

        valid = (
            size_ok
            and bottom_in_floor
            and touches_white
            and white_area >= self.ring_white_min_area
            and nonline_area >= self.ring_white_min_nonline_area
        )

        result = ring.copy()
        result["white_area"] = white_area
        result["white_nonline_area"] = nonline_area
        result["size_ok"] = size_ok
        result["touches_white"] = touches_white
        result["bottom_in_floor"] = bottom_in_floor
        result["valid"] = valid
        return result

    def validate_black_rings(self, rings, badminton_mask, floor_mask):
        validated = [
            self.validate_black_ring(ring, badminton_mask, floor_mask)
            for ring in rings
        ]
        validated.sort(
            key=lambda item: (
                not item["valid"],
                -item["white_area"],
            )
        )
        return validated

    def ring_to_candidate(self, ring):
        target_cx = ring["center_x"]
        target_cy = ring["center_y"]
        error_from_center = target_cx - self.width // 2
        return {
            "contour": ring["contour"],
            "rect": ring["rect"],
            "area": ring["area"],
            "rect_cx": target_cx,
            "rect_cy": target_cy,
            "target_cx": target_cx,
            "target_cy": target_cy,
            "error": error_from_center,
            "w_h_ratio": ring["ratio"],
            "head_found": True,
            "source": "ring",
            "white_area": ring["white_area"],
            "white_nonline_area": ring["white_nonline_area"],
        }

    def detect_black_ring_candidates(self, frame, badminton_mask, floor_mask):
        ring_mask = self.build_black_ring_mask(frame, floor_mask)
        rings = self.find_black_ring_candidates(ring_mask, floor_mask)
        validated = self.validate_black_rings(rings, badminton_mask, floor_mask)
        return [self.ring_to_candidate(ring) for ring in validated if ring["valid"]]

    def build_display_mask(self, floor_mask, badminton_mask):
        return badminton_mask

    def has_floor(self, floor_mask):
        return cv2.countNonZero(floor_mask) >= self.min_floor_area

    def detect_frame(self, frame):
        floor_mask = self.build_floor_mask(frame)
        badminton_mask = self.build_badminton_mask(frame)

        candidate = self.detect_black_ring_candidates(frame, badminton_mask, floor_mask)
        detection_mask = badminton_mask
        contours, _ = cv2.findContours(detection_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if not candidate:
            for cnt in contours:
                self.contour_dealing(cnt, candidate, None)

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
        if floor_mask is not None and self.has_floor(floor_mask) and floor_mask[rect_cy, rect_cx] == 0:
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
