from pathlib import Path
import sys

import cv2
import numpy as np

RPI_DIR = Path(__file__).resolve().parents[1]
if str(RPI_DIR) not in sys.path:
    sys.path.insert(0, str(RPI_DIR))

from camera_base import CameraBase


lower_white = np.array([0, 180, 0])
upper_white = np.array([180, 255, 40])
lower_white_hsv = np.array([0, 0, 180])
upper_white_hsv = np.array([180, 40, 255])
lower_floor = np.array([35, 40, 20])
upper_floor = np.array([95, 255, 180])

WINDOW_NAME = "Boundary Tuner"
TRACKBAR_WINDOW = "Boundary Controls"
WINDOW_POSITIONS = {
    WINDOW_NAME: (800, 150),
    TRACKBAR_WINDOW: (300, 250),
}
VIEW_MODES = ["original", "mask", "result", "floor_mask", "floor_result"]
MASK_VIEW_MODES = ["original", "mask", "result"]
FLOOR_VIEW_MODES = ["original", "floor_mask", "floor_result", "raw_mask"]
COLOR_SPACES = ["HLS", "HSV"]
CHANNEL_NAMES = {
    "HLS": ["H", "L", "S"],
    "HSV": ["H", "S", "V"],
}


def nothing(_):
    pass


def create_trackbars(color_space, lower, upper):
    channel_names = CHANNEL_NAMES[color_space]
    cv2.namedWindow(TRACKBAR_WINDOW)
    cv2.moveWindow(TRACKBAR_WINDOW, *WINDOW_POSITIONS[TRACKBAR_WINDOW])
    cv2.createTrackbar(f"{channel_names[0]} min", TRACKBAR_WINDOW, int(lower[0]), 180, nothing)
    cv2.createTrackbar(f"{channel_names[0]} max", TRACKBAR_WINDOW, int(upper[0]), 180, nothing)
    cv2.createTrackbar(f"{channel_names[1]} min", TRACKBAR_WINDOW, int(lower[1]), 255, nothing)
    cv2.createTrackbar(f"{channel_names[1]} max", TRACKBAR_WINDOW, int(upper[1]), 255, nothing)
    cv2.createTrackbar(f"{channel_names[2]} min", TRACKBAR_WINDOW, int(lower[2]), 255, nothing)
    cv2.createTrackbar(f"{channel_names[2]} max", TRACKBAR_WINDOW, int(upper[2]), 255, nothing)


def create_prefixed_trackbars(prefix, color_space, lower, upper):
    channel_names = CHANNEL_NAMES[color_space]
    cv2.createTrackbar(f"{prefix} {channel_names[0]} min", TRACKBAR_WINDOW, int(lower[0]), 180, nothing)
    cv2.createTrackbar(f"{prefix} {channel_names[0]} max", TRACKBAR_WINDOW, int(upper[0]), 180, nothing)
    cv2.createTrackbar(f"{prefix} {channel_names[1]} min", TRACKBAR_WINDOW, int(lower[1]), 255, nothing)
    cv2.createTrackbar(f"{prefix} {channel_names[1]} max", TRACKBAR_WINDOW, int(upper[1]), 255, nothing)
    cv2.createTrackbar(f"{prefix} {channel_names[2]} min", TRACKBAR_WINDOW, int(lower[2]), 255, nothing)
    cv2.createTrackbar(f"{prefix} {channel_names[2]} max", TRACKBAR_WINDOW, int(upper[2]), 255, nothing)


def get_bounds(color_space):
    channel_names = CHANNEL_NAMES[color_space]
    lower = np.array([
        cv2.getTrackbarPos(f"{channel_names[0]} min", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{channel_names[1]} min", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{channel_names[2]} min", TRACKBAR_WINDOW),
    ])
    upper = np.array([
        cv2.getTrackbarPos(f"{channel_names[0]} max", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{channel_names[1]} max", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{channel_names[2]} max", TRACKBAR_WINDOW),
    ])
    return lower, upper


def get_prefixed_bounds(prefix, color_space):
    channel_names = CHANNEL_NAMES[color_space]
    lower = np.array([
        cv2.getTrackbarPos(f"{prefix} {channel_names[0]} min", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{prefix} {channel_names[1]} min", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{prefix} {channel_names[2]} min", TRACKBAR_WINDOW),
    ])
    upper = np.array([
        cv2.getTrackbarPos(f"{prefix} {channel_names[0]} max", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{prefix} {channel_names[1]} max", TRACKBAR_WINDOW),
        cv2.getTrackbarPos(f"{prefix} {channel_names[2]} max", TRACKBAR_WINDOW),
    ])
    return lower, upper


def create_morph_trackbars(open_size=1, close_size=1):
    cv2.createTrackbar("open", TRACKBAR_WINDOW, int(open_size), 50, nothing)
    cv2.createTrackbar("close", TRACKBAR_WINDOW, int(close_size), 50, nothing)


def create_detail_trackbars():
    cv2.namedWindow(TRACKBAR_WINDOW)
    cv2.moveWindow(TRACKBAR_WINDOW, *WINDOW_POSITIONS[TRACKBAR_WINDOW])
    cv2.createTrackbar("ball HSV", TRACKBAR_WINDOW, 0, 1, nothing)
    create_prefixed_trackbars("ball HLS", "HLS", lower_white, upper_white)
    create_prefixed_trackbars("ball HSV", "HSV", lower_white_hsv, upper_white_hsv)
    create_prefixed_trackbars("floor", "HSV", lower_floor, upper_floor)
    cv2.createTrackbar("ball open", TRACKBAR_WINDOW, 3, 50, nothing)
    cv2.createTrackbar("ball close", TRACKBAR_WINDOW, 10, 50, nothing)
    cv2.createTrackbar("floor open", TRACKBAR_WINDOW, 5, 50, nothing)
    cv2.createTrackbar("floor close", TRACKBAR_WINDOW, 21, 50, nothing)


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


def build_badminton_mask(image, color_space, lower, upper):
    mask = build_mask(image, color_space, lower, upper)
    return apply_morphology(mask)


def build_badminton_mask_with_morph(image, color_space, lower, upper, open_size, close_size):
    mask = build_mask(image, color_space, lower, upper)
    if open_size > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if close_size > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    return mask


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

def detect_badminton_like_camera(image, badminton_mask, floor_mask, min_area_ratio):
    height, width = image.shape[:2]
    detector = CameraBase(width=width, height=height, flip_code=None)
    detector.min_floor_area = int(width * height * min_area_ratio)
    floor_found = detector.has_floor(floor_mask)
    display_mask = detector.build_display_mask(floor_mask, badminton_mask)
    contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate = []
    for cnt in contours:
        detector.contour_dealing(cnt, candidate, floor_mask)

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
        "HLS": [lower_white.copy(), upper_white.copy()],
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
        use_ball_hsv = cv2.getTrackbarPos("ball HSV", TRACKBAR_WINDOW) == 1
        color_space = "HSV" if use_ball_hsv else "HLS"
        ball_lower, ball_upper = get_prefixed_bounds(f"ball {color_space}", color_space)
        floor_lower, floor_upper = get_prefixed_bounds("floor", "HSV")
        ball_open = cv2.getTrackbarPos("ball open", TRACKBAR_WINDOW)
        ball_close = cv2.getTrackbarPos("ball close", TRACKBAR_WINDOW)
        floor_open = cv2.getTrackbarPos("floor open", TRACKBAR_WINDOW)
        floor_close = cv2.getTrackbarPos("floor close", TRACKBAR_WINDOW)
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
        floor_result = cv2.bitwise_and(image, image, mask=floor_mask)
        floor_contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(floor_result, floor_contours, -1, (255, 0, 255), 1)
        floor_found, find_ball, candidate, target, display_mask = detect_badminton_like_camera(
            image,
            badminton_mask,
            floor_mask,
            min_area_ratio,
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
            else:
                display = floor_result

        if view_mode in {"original", "result", "floor_result"}:
            cv2.drawContours(display, floor_contours, -1, (255, 0, 255), 1)
            for item in candidate:
                cv2.drawContours(display, [item["contour"]], -1, (0, 255, 0), 2)
            if find_ball and target is not None:
                cv2.circle(display, (target["target_cx"], target["target_cy"]), 6, (0, 0, 255), -1)

        extra_text = (
            f"ball_HSV={int(use_ball_hsv)}  ball_open={ball_open} ball_close={ball_close}  "
            f"floor_open={floor_open} floor_close={floor_close}"
        )
        display = draw_status(
            display,
            view_mode,
            color_space,
            ball_lower,
            ball_upper,
            help_text="v:view  p:print  q:quit",
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
            if use_ball_hsv:
                print("lower_white_hsv = np.array({})".format(ball_lower.tolist()))
                print("upper_white_hsv = np.array({})".format(ball_upper.tolist()))
            else:
                print("lower_white = np.array({})".format(ball_lower.tolist()))
                print("upper_white = np.array({})".format(ball_upper.tolist()))
            print(f"kernel_open = np.ones(({ball_open}, {ball_open}), np.uint8)")
            print(f"kernel_close = np.ones(({ball_close}, {ball_close}), np.uint8)")
            print("lower_floor = np.array({})".format(floor_lower.tolist()))
            print("upper_floor = np.array({})".format(floor_upper.tolist()))
            print(f"floor_kernel_open = np.ones(({floor_open}, {floor_open}), np.uint8)")
            print(f"floor_kernel_close = np.ones(({floor_close}, {floor_close}), np.uint8)")

    cv2.destroyAllWindows()




if __name__ == "__main__":
    i = 0
    #tune_floor_mask_single_image("stock", f"image_{i}.jpg")
    #tune_badminton_mask_single_image("stock", f"image_{i}.jpg")
    detail_single_alternation("stock", f"image_{i}.jpg")
