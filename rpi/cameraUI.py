import cv2
import numpy as np
import time
import logging
import os

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
    def __init__(self, width=320, height=240, camera_index=0):
        self.width = width
        self.height = height
        self.camera_index = camera_index
        #badminton
        self.lower_white = np.array([0, 180, 0])
        self.upper_white = np.array([180, 255, 40])
        self.lower_floor = np.array([35, 40, 20])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)
        #floor
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

        self.cap = cv2.VideoCapture(self.camera_index)
        self.closed = False

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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
        # deal with single frame
        find_ball = False
        error = None
        candidate = []

        floor_mask = self.build_floor_mask(frame)
        badminton_mask = self.build_badminton_mask(frame)
        contours, _ = cv2.findContours(badminton_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            self.contour_dealing(cnt, candidate, floor_mask)
        
        find_ball, error, target = self.choose_ball(candidate)

        self.visualize_frame(frame, floor_mask, candidate, target)
        return frame, floor_mask, badminton_mask, find_ball, error
    
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
    
    def contour_dealing(self, cnt, candidate, floor_mask):
            area = cv2.contourArea(cnt)
            if 800 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                (w, h) = rect[1]
                
                if w == 0 or h == 0: 
                    return
                
                ratio = min(w, h) / max(w, h)
                if ratio <= 0.35:
                    return
                
                cx, cy = int(rect[0][0]), int(rect[0][1])
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    return
                if floor_mask[cy, cx] == 0:
                    return

                error_from_center = cx - self.width // 2

                candidate.append({
                "contour": cnt,
                "rect": rect,
                "area": area,
                "cx": cx,
                "cy": cy,
                "error": error_from_center,
                "ratio": ratio
            })

    def draw_target(self, frame, target):
        cx = target["cx"]
        cy = target["cy"]

        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(frame,"TARGET",(cx - 25, cy + 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
        cv2.line(frame,(self.width // 2, cy),(cx, cy),(0, 255, 0),2)
        cv2.putText(frame,f"error={target['error']}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
        logger.debug("target area=%s cx=%s cy=%s ratio=%.3f",target["area"],cx,cy,target["ratio"])

    def visualize_frame(self, frame, floor_mask, candidate, target):
        self.draw_center_line(frame)
        self.draw_floor_boundary(frame, floor_mask)
        self.draw_candicate(frame, candidate)
        if target is not None:
            self.draw_target(frame, target)

    def draw_center_line(self, frame):
        cv2.line(frame,(self.width // 2, 0),(self.width // 2, self.height),(0, 255, 255),1)

    def draw_floor_boundary(self, frame, floor_mask):
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 0, 255), 3)

    def draw_candicate(self, frame, candidate):
        for ball in candidate:
            box = cv2.boxPoints(ball["rect"])
            box = np.intp(box)

            cx = ball["cx"]
            cy = ball["cy"]

            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            cv2.putText(frame,"Ball",(cx - 10, cy - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)

    def streaming(self):
        logger.info("Starting tracking... Press 'q' to quit.")

        at_frame = 0
        processed_frame = None
        floor_mask = None
        badminton_mask = None
        find_ball = False
        error = None

        try:
            while True:
                ret, raw_frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to read frame")
                    break

                raw_frame = cv2.resize(raw_frame, (self.width, self.height))

                if at_frame % 3 == 0:
                    processed_frame, floor_mask, badminton_mask, find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error

                if processed_frame is not None:
                    cv2.imshow("Robot View", processed_frame)

                if badminton_mask is not None and floor_mask is not None:
                    display_mask = cv2.bitwise_and(badminton_mask, badminton_mask, mask=floor_mask)
                    cv2.imshow("White Mask", display_mask)

                at_frame += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def single_test(self, filename="test_capture.jpg"):
        logger.info("capturing photo...")

        ret, raw_frame = self.cap.read()

        if not ret:
            logger.warning("Failed to capture photo")
            return

        raw_frame = cv2.resize(raw_frame, (self.width, self.height))

        processed_frame, floor_mask, badminton_mask, find_ball, error = self.process_frame(raw_frame)
        logger.info("find_ball=%s error=%s", find_ball, error)
        cv2.imwrite(filename, processed_frame)
        display_mask = cv2.bitwise_and(badminton_mask, badminton_mask, mask=floor_mask)
        cv2.imwrite(f"mask_{filename}", display_mask)

        logger.info("finish")

    def close(self):
        if self.closed:
            return
        self.cap.release()
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
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
