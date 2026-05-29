from pathlib import Path
import sys

import cv2
import numpy as np

RPI_DIR = Path(__file__).resolve().parents[1]
if str(RPI_DIR) not in sys.path:
    sys.path.insert(0, str(RPI_DIR))

from camera_base import CameraBase


lower_white_hls = np.array([0, 166, 0])
upper_white_hls = np.array([180, 255, 174])
lower_white_hsv = np.array([0, 0, 144])
upper_white_hsv = np.array([180, 53, 255])
lower_floor = np.array([35, 90, 35])
upper_floor = np.array([95, 255, 185])

WINDOW_NAME = "Boundary Tuner"
TRACKBAR_WINDOW = "Boundary Controls"
WINDOW_POSITIONS = {
    WINDOW_NAME: (650, 150),
    TRACKBAR_WINDOW: (200, 50),
}
TRACKBAR_WINDOW_SIZE = (360, 360)
DETAIL_TRACKBAR_WINDOW_SIZE = (360, 760)
VIEW_MODES = ["original", "mask", "result", "floor_mask", "floor_result", "ring_mask"]
MASK_VIEW_MODES = ["original", "mask", "result"]
FLOOR_VIEW_MODES = ["original", "floor_mask", "floor_result", "raw_mask"]
COLOR_SPACES = ["HLS", "HSV"]
CHANNEL_NAMES = {
    "HLS": ["H", "L", "S"],
    "HSV": ["H", "S", "V"],
}
TRACKBAR_PREFIXES = {
    "ball HLS": "ball",
    "ball HSV": "ball HSV",
    "floor": "F",
}
RING_FLOOR_TOP_BUFFER = 35
RING_ABOVE_FLOOR_TOP = 15
RING_BELOW_FLOOR_TOP = 55
RING_MIN_AREA = 20
RING_MAX_AREA = 1200
RING_MAX_SIZE = 70
RING_WHITE_ROI_X = 70
RING_WHITE_ROI_UP = 45
RING_WHITE_ROI_DOWN = 35
RING_WHITE_MIN_AREA = 120
RING_WHITE_MIN_NONLINE_AREA = 33
RING_MAX_HEIGHT = 50
RING_MAX_WIDTH = 50
RING_WHITE_NEAR_RADIUS = 8
RING_FLOOR_TOUCH_X = 5
RING_FLOOR_TOUCH_DOWN = 12
BOARD_IGNORE_LEFT_SIDE_Y_RATIO = 0.38
BOARD_IGNORE_LEFT_BOTTOM_X_RATIO = 0.32
BOARD_IGNORE_RIGHT_SIDE_Y_RATIO = 0.50
BOARD_IGNORE_RIGHT_BOTTOM_X_RATIO = 0.74

def nothing(_):
    pass


def setup_trackbar_window(window_name=TRACKBAR_WINDOW, size=TRACKBAR_WINDOW_SIZE):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *size)
    cv2.moveWindow(window_name, *WINDOW_POSITIONS[window_name])


def channel_trackbar_name(channel_name, is_lower, prefix=None):
    short_prefix = TRACKBAR_PREFIXES.get(prefix, prefix)
    suffix = "min" if is_lower else "max"
    if short_prefix:
        return f"{short_prefix} {channel_name} {suffix}"
    return f"{channel_name} {suffix}"


def create_trackbars(color_space, lower, upper):
    channel_names = CHANNEL_NAMES[color_space]
    setup_trackbar_window()
    cv2.createTrackbar(channel_trackbar_name(channel_names[0], True), TRACKBAR_WINDOW, int(lower[0]), 180, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[0], False), TRACKBAR_WINDOW, int(upper[0]), 180, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[1], True), TRACKBAR_WINDOW, int(lower[1]), 255, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[1], False), TRACKBAR_WINDOW, int(upper[1]), 255, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[2], True), TRACKBAR_WINDOW, int(lower[2]), 255, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[2], False), TRACKBAR_WINDOW, int(upper[2]), 255, nothing)


def create_prefixed_trackbars(prefix, color_space, lower, upper, window_name=TRACKBAR_WINDOW):
    channel_names = CHANNEL_NAMES[color_space]
    cv2.createTrackbar(channel_trackbar_name(channel_names[0], True, prefix), window_name, int(lower[0]), 180, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[0], False, prefix), window_name, int(upper[0]), 180, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[1], True, prefix), window_name, int(lower[1]), 255, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[1], False, prefix), window_name, int(upper[1]), 255, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[2], True, prefix), window_name, int(lower[2]), 255, nothing)
    cv2.createTrackbar(channel_trackbar_name(channel_names[2], False, prefix), window_name, int(upper[2]), 255, nothing)


def get_bounds(color_space):
    channel_names = CHANNEL_NAMES[color_space]
    lower = np.array([
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[0], True), TRACKBAR_WINDOW),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[1], True), TRACKBAR_WINDOW),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[2], True), TRACKBAR_WINDOW),
    ])
    upper = np.array([
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[0], False), TRACKBAR_WINDOW),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[1], False), TRACKBAR_WINDOW),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[2], False), TRACKBAR_WINDOW),
    ])
    return lower, upper


def get_prefixed_bounds(prefix, color_space, window_name=TRACKBAR_WINDOW):
    channel_names = CHANNEL_NAMES[color_space]
    lower = np.array([
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[0], True, prefix), window_name),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[1], True, prefix), window_name),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[2], True, prefix), window_name),
    ])
    upper = np.array([
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[0], False, prefix), window_name),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[1], False, prefix), window_name),
        cv2.getTrackbarPos(channel_trackbar_name(channel_names[2], False, prefix), window_name),
    ])
    return lower, upper


def create_morph_trackbars(open_size=1, close_size=1):
    cv2.createTrackbar("open", TRACKBAR_WINDOW, int(open_size), 50, nothing)
    cv2.createTrackbar("close", TRACKBAR_WINDOW, int(close_size), 50, nothing)


def create_detail_trackbars():
    setup_trackbar_window(TRACKBAR_WINDOW, DETAIL_TRACKBAR_WINDOW_SIZE)
    create_prefixed_trackbars("ball HLS", "HLS", lower_white_hls, upper_white_hls)
    create_prefixed_trackbars("floor", "HSV", lower_floor, upper_floor)
    cv2.createTrackbar("ball open", TRACKBAR_WINDOW, 3, 50, nothing)
    cv2.createTrackbar("ball close", TRACKBAR_WINDOW, 10, 50, nothing)
    cv2.createTrackbar("F open", TRACKBAR_WINDOW, 5, 50, nothing)
    cv2.createTrackbar("F close", TRACKBAR_WINDOW, 21, 50, nothing)
    cv2.createTrackbar("ring V max", TRACKBAR_WINDOW, 80, 255, nothing)
    cv2.createTrackbar("line min", TRACKBAR_WINDOW, RING_WHITE_MIN_NONLINE_AREA, 1000, nothing)


def get_morph_kernel(name):
    size = cv2.getTrackbarPos(name, TRACKBAR_WINDOW)
    if size <= 0:
        return None
    return np.ones((size, size), np.uint8)


def apply_morphology(mask):
    open_kernel = get_morph_kernel("open")
    close_kernel = get_morph_kernel("close")

    if open_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    if close_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def build_mask(image, color_space, lower, upper):
    if color_space == "HLS":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(converted, lower, upper)


def board_ignore_polygons(height, width):
    left_side_y = int(height * BOARD_IGNORE_LEFT_SIDE_Y_RATIO)
    left_bottom_x = int(width * BOARD_IGNORE_LEFT_BOTTOM_X_RATIO)
    right_side_y = int(height * BOARD_IGNORE_RIGHT_SIDE_Y_RATIO)
    right_bottom_x = int(width * BOARD_IGNORE_RIGHT_BOTTOM_X_RATIO)
    return [
        np.array([(0, height - 1), (0, left_side_y), (left_bottom_x, height - 1)], dtype=np.int32),
        np.array(
            [(width - 1, height - 1), (width - 1, right_side_y), (right_bottom_x, height - 1)],
            dtype=np.int32,
        ),
    ]


def build_roi_keep_mask(height, width):
    keep_mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.fillPoly(keep_mask, board_ignore_polygons(height, width), 0)
    return keep_mask


def apply_roi_keep_mask(mask):
    height, width = mask.shape[:2]
    return cv2.bitwise_and(mask, mask, mask=build_roi_keep_mask(height, width))


def draw_board_ignore_polygons(image):
    height, width = image.shape[:2]
    for polygon in board_ignore_polygons(height, width):
        cv2.polylines(image, [polygon], True, (0, 255, 255), 1)


def build_badminton_mask(image, color_space, lower, upper):
    mask = build_mask(image, color_space, lower, upper)
    return apply_roi_keep_mask(apply_morphology(mask))


def build_badminton_mask_with_morph(image, color_space, lower, upper, open_size, close_size):
    mask = build_mask(image, color_space, lower, upper)
    if open_size > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if close_size > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    return apply_roi_keep_mask(mask)


def build_floor_top_search_mask(floor_mask, top_buffer=RING_FLOOR_TOP_BUFFER):
    height, width = floor_mask.shape[:2]
    if cv2.countNonZero(floor_mask) == 0:
        return np.full((height, width), 255, dtype=np.uint8)

    search_mask = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        floor_y = np.flatnonzero(floor_mask[:, x])
        if len(floor_y) == 0:
            continue
        top_y = max(int(floor_y[0]) - top_buffer, 0)
        search_mask[top_y:, x] = 255
    return search_mask


def build_black_ring_mask(image, floor_mask, v_max, open_size, close_size):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ring_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, v_max]))
    search_mask = build_floor_top_search_mask(floor_mask)
    ring_mask = cv2.bitwise_and(ring_mask, ring_mask, mask=search_mask)

    if open_size > 0:
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if close_size > 0:
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    return apply_roi_keep_mask(ring_mask)


def is_near_floor_top(floor_mask, x, y, above=RING_ABOVE_FLOOR_TOP, below=RING_BELOW_FLOOR_TOP):
    if floor_mask is None or cv2.countNonZero(floor_mask) == 0:
        return True
    if not (0 <= x < floor_mask.shape[1]):
        return False

    floor_y = np.flatnonzero(floor_mask[:, x])
    if len(floor_y) == 0:
        return False

    top_y = int(floor_y[0])
    return top_y - above <= y <= top_y + below


def find_black_ring_candidates(ring_mask, floor_mask=None):
    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (RING_MIN_AREA <= area <= RING_MAX_AREA):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w > RING_MAX_SIZE or h > RING_MAX_SIZE:
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
        if not is_near_floor_top(floor_mask, center_x, center_y):
            continue

        candidates.append({
            "contour": cnt,
            "area": area,
            "bbox": (x, y, w, h),
            "center_x": center_x,
            "center_y": center_y,
            "ratio": ratio,
        })

    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates


def crop_ring_white_roi(mask, ring):
    height, width = mask.shape[:2]
    cx = ring["center_x"]
    cy = ring["center_y"]
    x1 = max(cx - RING_WHITE_ROI_X, 0)
    x2 = min(cx + RING_WHITE_ROI_X, width)
    y1 = max(cy - RING_WHITE_ROI_UP, 0)
    y2 = min(cy + RING_WHITE_ROI_DOWN, height)
    return mask[y1:y2, x1:x2], (x1, y1, x2, y2)


def horizontal_line_stats(mask):
    white_area = cv2.countNonZero(mask)
    if white_area == 0:
        return 0, 0

    kernel_width = max(mask.shape[1] // 3, 15)
    horizontal_kernel = np.ones((1, kernel_width), np.uint8)
    horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    line_area = cv2.countNonZero(horizontal_lines)
    return line_area / white_area, white_area - line_area


def ring_bottom_in_floor(ring, floor_mask):
    if floor_mask is None or cv2.countNonZero(floor_mask) == 0:
        return True

    x, y, w, h = ring["bbox"]
    bottom_x = x + w // 2
    bottom_y = y + h - 1

    height, width = floor_mask.shape[:2]
    if not (0 <= bottom_x < width and 0 <= bottom_y < height):
        return False

    x1 = max(bottom_x - RING_FLOOR_TOUCH_X, 0)
    x2 = min(bottom_x + RING_FLOOR_TOUCH_X + 1, width)
    y1 = bottom_y
    y2 = min(bottom_y + RING_FLOOR_TOUCH_DOWN + 1, height)
    floor_patch = floor_mask[y1:y2, x1:x2]
    return cv2.countNonZero(floor_patch) > 0


def validate_ring_with_white_mask(ring, badminton_mask, floor_mask, min_white_area, min_nonline_area):
    roi, roi_box = crop_ring_white_roi(badminton_mask, ring)
    white_area = cv2.countNonZero(roi)
    line_ratio, nonline_area = horizontal_line_stats(roi)
    x, y, w, h = ring["bbox"]
    size_ok = w <= RING_MAX_WIDTH and h <= RING_MAX_HEIGHT

    near_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (RING_WHITE_NEAR_RADIUS * 2 + 1, RING_WHITE_NEAR_RADIUS * 2 + 1),
    )
    near_white = cv2.dilate(badminton_mask, near_kernel)
    touches_white = cv2.countNonZero(near_white[y:y + h, x:x + w]) > 0
    bottom_in_floor = ring_bottom_in_floor(ring, floor_mask)

    valid = (
        size_ok
        and bottom_in_floor
        and touches_white
        and white_area >= min_white_area
        and nonline_area >= min_nonline_area
    )

    result = ring.copy()
    result["white_area"] = white_area
    result["white_line_ratio"] = line_ratio
    result["white_nonline_area"] = nonline_area
    result["white_roi"] = roi_box
    result["size_ok"] = size_ok
    result["bottom_in_floor"] = bottom_in_floor
    result["touches_white"] = touches_white
    result["valid"] = valid
    return result


def validate_ring_candidates_with_white(ring_candidates, badminton_mask, floor_mask, min_white_area, min_nonline_area):
    validated = [
        validate_ring_with_white_mask(ring, badminton_mask, floor_mask, min_white_area, min_nonline_area)
        for ring in ring_candidates
    ]
    validated.sort(
        key=lambda item: (
            not item["valid"],
            -item["white_area"],
            item["white_line_ratio"],
        )
    )
    return validated


def build_floor_mask(image, lower, upper, open_size, close_size, min_area_ratio, bottom_band_ratio, margin):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, lower, upper)

    floor_mask = raw_mask.copy()
    if open_size > 0:
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if close_size > 0:
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))

    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros((height, width), dtype=np.uint8)
    min_floor_area = int(width * height * min_area_ratio)
    bottom_y = int(height * bottom_band_ratio)
    floor_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        _, y, _, h = cv2.boundingRect(cnt)
        reaches_bottom_band = y + h >= bottom_y
        if area >= min_floor_area and reaches_bottom_band:
            floor_contours.append(cnt)

    if floor_contours:
        floor_hulls = [cv2.convexHull(cnt) for cnt in floor_contours]
        cv2.drawContours(filtered_mask, floor_hulls, -1, 255, -1)

    if margin > 0:
        filtered_mask = cv2.erode(filtered_mask, np.ones((margin, margin), np.uint8))

    return raw_mask, filtered_mask

def detect_badminton_like_camera(image, badminton_mask, floor_mask, min_area_ratio, use_floor_mask=True):
    height, width = image.shape[:2]
    detector = CameraBase(width=width, height=height, flip_code=None)
    detector.min_floor_area = int(width * height * min_area_ratio)
    floor_found = detector.has_floor(floor_mask)
    if use_floor_mask:
        display_mask = detector.build_display_mask(floor_mask, badminton_mask)
        position_mask = floor_mask
    else:
        display_mask = badminton_mask
        position_mask = np.zeros((height, width), dtype=np.uint8)
    contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate = []
    for cnt in contours:
        detector.contour_dealing(cnt, candidate, position_mask)

    find_ball, _, target = detector.choose_ball(candidate)
    return floor_found, find_ball, candidate, target, display_mask


def draw_status(
    image,
    view_mode,
    color_space,
    lower,
    upper,
    help_text="v:view  m:color  p:print  q:quit",
    extra_text=None,
    show_bounds=True,
):
    font_scale = 0.45
    line_height = 22
    if show_bounds:
        status_lines = [
            f"{color_space} {view_mode}  lower={lower.tolist()} upper={upper.tolist()}",
            help_text,
        ]
    else:
        status_lines = [f"{color_space} {view_mode}", help_text]
    if extra_text:
        status_lines.append(extra_text)
    status_height = line_height * len(status_lines) + 10
    status_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for line in status_lines
    )
    display_width = max(image.shape[1], status_width + 20)
    image_x = (display_width - image.shape[1]) // 2
    display = np.zeros((image.shape[0] + status_height, display_width, 3), dtype=np.uint8)
    display[:image.shape[0], image_x:image_x + image.shape[1]] = image
    status_y = image.shape[0] + 18

    for index, line in enumerate(status_lines):
        cv2.putText(
            display,
            line,
            (10, status_y + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            1,
        )
    return display


def print_bounds(color_space, lower, upper):
    lower_name = f"lower_{color_space.lower()}"
    upper_name = f"upper_{color_space.lower()}"
    print(f"{lower_name} = np.array({lower.tolist()})")
    print(f"{upper_name} = np.array({upper.tolist()})")


def print_badminton_settings(color_space, lower, upper):
    print_bounds(color_space, lower, upper)
    print(f"open_size = {cv2.getTrackbarPos('open', TRACKBAR_WINDOW)}")
    print(f"close_size = {cv2.getTrackbarPos('close', TRACKBAR_WINDOW)}")


def print_floor_settings(lower, upper):
    print("lower_floor = np.array({})".format(lower.tolist()))
    print("upper_floor = np.array({})".format(upper.tolist()))
    print(f"floor_kernel_open = {cv2.getTrackbarPos('open', TRACKBAR_WINDOW)}")
    print(f"floor_kernel_close = {cv2.getTrackbarPos('close', TRACKBAR_WINDOW)}")
    print(f"min_floor_area_ratio = {cv2.getTrackbarPos('min area %', TRACKBAR_WINDOW) / 100}")
    print(f"floor_bottom_band_ratio = {cv2.getTrackbarPos('bottom band %', TRACKBAR_WINDOW) / 100}")
    print(f"floor_boundary_margin = {cv2.getTrackbarPos('margin', TRACKBAR_WINDOW)}")


def tune_badminton_mask_single_image(dir, image_file):
    image_path = Path(dir) / image_file
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"{image_path} not found")
        return

    view_index = 0
    color_index = 0
    bounds_by_space = {
        "HLS": [lower_white_hls.copy(), upper_white_hls.copy()],
        "HSV": [np.array([0, 0, 180]), np.array([180, 40, 255])],
    }

    create_trackbars(
        COLOR_SPACES[color_index],
        bounds_by_space[COLOR_SPACES[color_index]][0],
        bounds_by_space[COLOR_SPACES[color_index]][1],
    )
    create_morph_trackbars(open_size=3, close_size=10)
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, *WINDOW_POSITIONS[WINDOW_NAME])

    while True:
        view_mode = MASK_VIEW_MODES[view_index]
        color_space = COLOR_SPACES[color_index]
        lower, upper = get_bounds(color_space)
        bounds_by_space[color_space] = [lower, upper]
        mask = build_badminton_mask(image, color_space, lower, upper)

        if view_mode == "original":
            display = image
        elif view_mode == "mask":
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = cv2.bitwise_and(image, image, mask=mask)

        display = draw_status(display, view_mode, color_space, lower, upper)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            view_index = (view_index + 1) % len(MASK_VIEW_MODES)
        if key == ord("m"):
            color_index = (color_index + 1) % len(COLOR_SPACES)
            color_space = COLOR_SPACES[color_index]
            cv2.destroyWindow(TRACKBAR_WINDOW)
            create_trackbars(color_space, bounds_by_space[color_space][0], bounds_by_space[color_space][1])
            create_morph_trackbars(open_size=3, close_size=10)
        if key == ord("p"):
            print_badminton_settings(color_space, lower, upper)

    cv2.destroyAllWindows()


def tune_floor_mask_single_image(dir, image_file):
    image_path = Path(dir) / image_file
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"{image_path} not found")
        return

    view_index = 0
    create_trackbars("HSV", lower_floor, upper_floor)
    create_morph_trackbars(open_size=5, close_size=21)
    cv2.createTrackbar("min area %", TRACKBAR_WINDOW, 3, 100, nothing)
    cv2.createTrackbar("bottom band %", TRACKBAR_WINDOW, 75, 100, nothing)
    cv2.createTrackbar("margin", TRACKBAR_WINDOW, 0, 50, nothing)
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, *WINDOW_POSITIONS[WINDOW_NAME])

    while True:
        view_mode = FLOOR_VIEW_MODES[view_index]
        lower, upper = get_bounds("HSV")
        open_size = cv2.getTrackbarPos("open", TRACKBAR_WINDOW)
        close_size = cv2.getTrackbarPos("close", TRACKBAR_WINDOW)
        min_area_ratio = cv2.getTrackbarPos("min area %", TRACKBAR_WINDOW) / 100
        bottom_band_ratio = cv2.getTrackbarPos("bottom band %", TRACKBAR_WINDOW) / 100
        margin = cv2.getTrackbarPos("margin", TRACKBAR_WINDOW)
        raw_mask, floor_mask = build_floor_mask(
            image,
            lower,
            upper,
            open_size,
            close_size,
            min_area_ratio,
            bottom_band_ratio,
            margin,
        )
        floor_result = cv2.bitwise_and(image, image, mask=floor_mask)
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(floor_result, contours, -1, (255, 0, 255), 1)

        if view_mode == "original":
            display = image
        elif view_mode == "floor_mask":
            display = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2BGR)
        elif view_mode == "raw_mask":
            display = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
        else:
            display = floor_result

        display = draw_status(display, view_mode, "HSV", lower, upper, help_text="v:view  p:print  q:quit")
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            view_index = (view_index + 1) % len(FLOOR_VIEW_MODES)
        if key == ord("p"):
            print_floor_settings(lower, upper)

    cv2.destroyAllWindows()


def detail_single_alternation(dir, image_file):
    image_path = Path(dir) / image_file
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"{image_path} not found")
        return

    view_index = 0
    create_detail_trackbars()
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, *WINDOW_POSITIONS[WINDOW_NAME])

    while True:
        view_mode = VIEW_MODES[view_index]
        color_space = "HLS"
        ball_lower, ball_upper = get_prefixed_bounds(f"ball {color_space}", color_space)
        floor_lower, floor_upper = get_prefixed_bounds("floor", "HSV")
        ball_open = cv2.getTrackbarPos("ball open", TRACKBAR_WINDOW)
        ball_close = cv2.getTrackbarPos("ball close", TRACKBAR_WINDOW)
        floor_open = cv2.getTrackbarPos("F open", TRACKBAR_WINDOW)
        floor_close = cv2.getTrackbarPos("F close", TRACKBAR_WINDOW)
        ring_v_max = cv2.getTrackbarPos("ring V max", TRACKBAR_WINDOW)
        ring_open = 1
        ring_close = 3
        ring_white_min = RING_WHITE_MIN_AREA
        ring_nonline_min = cv2.getTrackbarPos("line min", TRACKBAR_WINDOW)
        min_area_ratio = 0.03
        bottom_band_ratio = 0.75
        margin = 0
        badminton_mask = build_badminton_mask_with_morph(
            image,
            color_space,
            ball_lower,
            ball_upper,
            ball_open,
            ball_close,
        )
        _, floor_mask = build_floor_mask(
            image,
            floor_lower,
            floor_upper,
            floor_open,
            floor_close,
            min_area_ratio,
            bottom_band_ratio,
            margin,
        )
        ring_mask = build_black_ring_mask(
            image,
            floor_mask,
            ring_v_max,
            ring_open,
            ring_close,
        )
        ring_candidates = find_black_ring_candidates(ring_mask, floor_mask)
        validated_rings = validate_ring_candidates_with_white(
            ring_candidates,
            badminton_mask,
            floor_mask,
            ring_white_min,
            ring_nonline_min,
        )
        valid_rings = [ring for ring in validated_rings if ring["valid"]]
        ring_target = valid_rings[0] if valid_rings else None
        floor_result = cv2.bitwise_and(image, image, mask=floor_mask)
        floor_contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(floor_result, floor_contours, -1, (255, 0, 255), 1)
        floor_found, find_ball, candidate, target, display_mask = detect_badminton_like_camera(
            image,
            badminton_mask,
            floor_mask,
            min_area_ratio,
            use_floor_mask=False,
        )

        if view_mode == "original":
            display = image.copy()
        elif view_mode == "mask":
            display = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
        elif view_mode == "result":
            display = cv2.bitwise_and(image, image, mask=display_mask)
        else:
            if view_mode == "floor_mask":
                display = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2BGR)
            elif view_mode == "ring_mask":
                display = cv2.cvtColor(ring_mask, cv2.COLOR_GRAY2BGR)
            else:
                display = floor_result

        if view_mode in {"original", "result", "floor_result", "ring_mask"}:
            draw_board_ignore_polygons(display)
            cv2.drawContours(display, floor_contours, -1, (255, 0, 255), 1)
            for ring in validated_rings:
                x, y, w, h = ring["bbox"]
                color = (0, 255, 0) if ring["valid"] else (255, 0, 0)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.circle(display, (ring["center_x"], ring["center_y"]), 3, (255, 255, 0), -1)
                rx1, ry1, rx2, ry2 = ring["white_roi"]
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 1)
            for item in candidate:
                cv2.drawContours(display, [item["contour"]], -1, (0, 255, 0), 2)
            if ring_target is not None:
                cv2.circle(display, (ring_target["center_x"], ring_target["center_y"]), 6, (0, 0, 255), -1)

        extra_text = (
            f"ball_open={ball_open} ball_close={ball_close}  "
            f"floor_open={floor_open} floor_close={floor_close}  "
            f"ring_V={ring_v_max} ring_ok={len(valid_rings)}/{len(validated_rings)} "
            f"nonline_min={ring_nonline_min}"
        )
        display = draw_status(
            display,
            view_mode,
            color_space,
            ball_lower,
            ball_upper,
            help_text="v:view  p:print  d:debug  q:quit",
            extra_text=extra_text,
            show_bounds=False,
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            view_index = (view_index + 1) % len(VIEW_MODES)
        if key == ord("p"):
            print(f"ball_color_space = {color_space}")
            print("lower_white_hls = np.array({})".format(ball_lower.tolist()))
            print("upper_white_hls = np.array({})".format(ball_upper.tolist()))
            print(f"kernel_open = np.ones(({ball_open}, {ball_open}), np.uint8)")
            print(f"kernel_close = np.ones(({ball_close}, {ball_close}), np.uint8)")
            print("lower_floor = np.array({})".format(floor_lower.tolist()))
            print("upper_floor = np.array({})".format(floor_upper.tolist()))
            print(f"floor_kernel_open = np.ones(({floor_open}, {floor_open}), np.uint8)")
            print(f"floor_kernel_close = np.ones(({floor_close}, {floor_close}), np.uint8)")
            print(f"ring_v_max = {ring_v_max}")
            print(f"ring_open = {ring_open}")
            print(f"ring_close = {ring_close}")
            print(f"ring_white_min = {ring_white_min}")
            print(f"ring_nonline_min = {ring_nonline_min}")
            print(f"ring_count = {len(validated_rings)}")
            print(f"valid_ring_count = {len(valid_rings)}")
            if ring_target is not None:
                print(f"ring_target = ({ring_target['center_x']}, {ring_target['center_y']})")
        if key == ord("d"):
            for index, ring in enumerate(validated_rings, start=1):
                print(
                    "ring_{}: valid={} bbox={} center=({}, {}) area={:.1f} ratio={:.2f} "
                    "size_ok={} bottom_in_floor={} touches_white={} "
                    "white_area={} nonline_area={} line_ratio={:.2f}".format(
                        index,
                        ring["valid"],
                        ring["bbox"],
                        ring["center_x"],
                        ring["center_y"],
                        ring["area"],
                        ring["ratio"],
                        ring["size_ok"],
                        ring["bottom_in_floor"],
                        ring["touches_white"],
                        ring["white_area"],
                        ring["white_nonline_area"],
                        ring["white_line_ratio"],
                    )
                )

    cv2.destroyAllWindows()




if __name__ == "__main__":
    i = 69
    #tune_floor_mask_single_image("stock", f"image_{i}.jpg")
    #tune_badminton_mask_single_image("stock", f"image_{i}.jpg")
    detail_single_alternation("stock", f"image_{i}.jpg")
