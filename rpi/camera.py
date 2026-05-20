import cv2
import numpy as np
import time
import logging
import os
from picamera2 import Picamera2

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


class Camera:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        #badminton
        self.lower_white = np.array([0, 180, 0])
        self.upper_white = np.array([180, 255, 40])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)
        #floor
        self.lower_floor = np.array([35, 40, 20])
        self.upper_floor = np.array([95, 255, 180])
        self.floor_kernel_open = np.ones((5, 5), np.uint8)
        self.floor_kernel_close = np.ones((21, 21), np.uint8)
        self.floor_boundary_margin = 0
        self.floor_bottom_band_ratio = 0.75
        self.min_floor_area = int(self.width * self.height * 0.03)
        #target
        self.target_x = None
        self.target_y = None
        self.last_error = None
        self.lost_count = 0
        self.max_lost_frames = 5
        self.max_tracking_distance = 50

        self.picam2 = Picamera2()
        self.closed = False
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"AwbMode": 0})
        
        logger.info("camera activating...")
        time.sleep(2)

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

    def process_frame(self, frame):
        #deal with single frame        
        candidate = []
        floor_mask = self.build_floor_mask(frame)
        badminton_mask = self.build_badminton_mask(frame)
        contours, _ = cv2.findContours(badminton_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            self.contour_dealing(cnt, candidate, floor_mask)
        
        find_ball, error, target = self.choose_ball(candidate)

        return frame, badminton_mask, find_ball, error

    def contour_dealing(self, cnt, candidate, floor_mask):
        area = cv2.contourArea(cnt)
        if 800 < area < 10000:
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            
            if w == 0 or h == 0: 
                return
            
            ratio = min(w, h) / max(w, h)
            if ratio > 0.35:
                cx, cy = int(rect[0][0]), int(rect[0][1])
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    return
                if floor_mask[cy, cx] == 0:
                    return

                error_from_center = cx - self.width //2
                candidate.append({
                "contour": cnt,
                "rect": rect,
                "area": area,
                "cx": cx,
                "cy": cy,
                "error": error_from_center,
                "ratio": ratio
            })

    def choose_ball(self, candidate):
        if len(candidate) > 0:
            if self.target_x is None:
                target = min(candidate, key=lambda b: abs(b["error"]))
                distance = 0
            else:
                target = min(candidate,key=lambda b: (b["cx"] - self.target_x) ** 2 + (b["cy"] - self.target_y) ** 2)
                distance = ((target["cx"] - self.target_x) ** 2 +(target["cy"] - self.target_y) ** 2) ** 0.5

            if self.target_x is not None and distance > self.max_tracking_distance:
                self.lost_count += 1
                if self.lost_count <= self.max_lost_frames:
                    return True, self.last_error, None
                else:
                    self.target_x = None
                    self.target_y = None
                    self.last_error = None
                    return False, None, None

            self.target_x = target["cx"]
            self.target_y = target["cy"]
            error = target["error"]
            self.last_error = error
            self.lost_count = 0

            return True, error, target

        else:
            self.lost_count += 1

            if self.last_error is not None and self.lost_count <= self.max_lost_frames:
                return True, self.last_error, None
            else:
                self.target_x = None
                self.target_y = None
                self.last_error = None
                return False, None, None

    def streaming(self):
        logger.info("Starting tracking... Press 'q' to quit.")
        at_frame = 0
        processed_frame = None
        badminton_mask = None
        find_ball = False
        error = None
        try:
            while True:
                raw_frame = self.picam2.capture_array()
                if at_frame % 3 == 0:
                    processed_frame, badminton_mask, find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error            
                at_frame += 1
                if processed_frame is not None:
                    cv2.imshow('Robot View', processed_frame)
                if badminton_mask is not None:
                    cv2.imshow('White Mask', badminton_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.close()
        
    def single_test(self, filename="test_capture.jpg"):
        logger.info("capturing photo...")
        raw_frame = self.picam2.capture_array()
        processed_frame, mask, find_ball, error = self.process_frame(raw_frame)
        logger.info("find_ball=%s error=%s", find_ball, error)
        cv2.imwrite(f"/home/waryt/Desktop/{filename}", processed_frame)
        cv2.imwrite(f"/home/waryt/Desktop/mask_{filename}", mask)
        logger.info("finish")
    
    def close(self):
        if self.closed:
            return
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.closed = True
        logger.info("Camera and Windows closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        tracker.single_test()        
        #tracker.streaming()
