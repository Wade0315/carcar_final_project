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

WINDOW_NAME = "Boundary Tuner"
TRACKBAR_WINDOW = "Boundary Controls"
WINDOW_POSITIONS = {
    WINDOW_NAME: (800, 150),
    TRACKBAR_WINDOW: (300, 250),
}
VIEW_MODES = ["original", "mask", "result", "floor_mask", "floor_result"]
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


def build_mask(image, color_space, lower, upper):
    if color_space == "HLS":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(converted, lower, upper)


def build_floor_views(image):
    height, width = image.shape[:2]
    detector = CameraBase(width=width, height=height, flip_code=None)
    floor_mask = detector.build_floor_mask(image)
    floor_result = cv2.bitwise_and(image, image, mask=floor_mask)

    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(floor_result, contours, -1, (255, 0, 255), 3)
    return floor_mask, floor_result


def draw_status(image, view_mode, color_space, lower, upper):
    display = image.copy()
    cv2.putText(
        display,
        f"{color_space} {view_mode}  lower={lower.tolist()} upper={upper.tolist()}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        display,
        "v:view  m:color  p:print  q:quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return display


def print_bounds(color_space, lower, upper):
    lower_name = f"lower_{color_space.lower()}"
    upper_name = f"upper_{color_space.lower()}"
    print(f"{lower_name} = np.array({lower.tolist()})")
    print(f"{upper_name} = np.array({upper.tolist()})")


def detail_single_alternation(dir, image_file):
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
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, *WINDOW_POSITIONS[WINDOW_NAME])

    while True:
        view_mode = VIEW_MODES[view_index]
        color_space = COLOR_SPACES[color_index]
        lower, upper = get_bounds(color_space)
        bounds_by_space[color_space] = [lower, upper]
        mask = build_mask(image, color_space, lower, upper)

        if view_mode == "original":
            display = image
        elif view_mode == "mask":
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif view_mode == "result":
            display = cv2.bitwise_and(image, image, mask=mask)
        else:
            floor_mask, floor_result = build_floor_views(image)
            if view_mode == "floor_mask":
                display = cv2.cvtColor(floor_mask, cv2.COLOR_GRAY2BGR)
            else:
                display = floor_result

        display = draw_status(display, view_mode, color_space, lower, upper)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            view_index = (view_index + 1) % len(VIEW_MODES)
        if key == ord("m"):
            color_index = (color_index + 1) % len(COLOR_SPACES)
            color_space = COLOR_SPACES[color_index]
            cv2.destroyWindow(TRACKBAR_WINDOW)
            create_trackbars(color_space, bounds_by_space[color_space][0], bounds_by_space[color_space][1])
        if key == ord("p"):
            print_bounds(color_space, lower, upper)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detail_single_alternation("stock", "test_image.jpg")
