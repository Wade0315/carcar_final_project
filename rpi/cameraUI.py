import cv2
import numpy as np
import time


class Camera:
    def __init__(self, width=320, height=240, camera_index=0):
        self.width = width
        self.height = height
        self.camera_index = camera_index

        self.lower_white = np.array([0, 180, 0])
        self.upper_white = np.array([180, 255, 40])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)

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

        print("camera activating...")
        time.sleep(2)

    def process_frame(self, frame):
        # deal with single frame
        find_ball = False
        error = None
        candidate = []

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, self.lower_white, self.upper_white)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            self.contour_dealing(cnt, candidate)
        
        find_ball, error, target = self.choose_ball(candidate)

        #visualization
        self.draw_center_line(frame)
        self.draw_candicate(frame, candidate)
        if target is not None:
            self.draw_target(frame, target)
        return frame, mask, find_ball, error
    
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
    
    def contour_dealing(self, cnt, candidate):
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
        print(f"area: {target['area']}, cx: {cx}, cy: {cy}, ratio: {target['ratio']:.3f}")
    def draw_center_line(self, frame):
        cv2.line(frame,(self.width // 2, 0),(self.width // 2, self.height),(0, 255, 255),1)

    def draw_candicate(self, frame, candidate):
        for ball in candidate:
            box = cv2.boxPoints(ball["rect"])
            box = np.intp(box)

            cx = ball["cx"]
            cy = ball["cy"]

            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            cv2.putText(frame,"Ball",(cx - 10, cy - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)

    def streaming(self):
        print("Starting tracking... Press 'q' to quit.")

        at_frame = 0
        processed_frame = None
        mask = None
        find_ball = False
        error = None

        try:
            while True:
                ret, raw_frame = self.cap.read()

                if not ret:
                    print("Failed to read frame")
                    break

                raw_frame = cv2.resize(raw_frame, (self.width, self.height))

                if at_frame % 3 == 0:
                    processed_frame, mask, find_ball, error = self.process_frame(raw_frame)

                yield find_ball, error

                if processed_frame is not None:
                    cv2.imshow("Robot View", processed_frame)

                if mask is not None:
                    cv2.imshow("White Mask", mask)

                at_frame += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            self.close()

    def single_test(self, filename="test_capture.jpg"):
        print("capturing photo...")

        ret, raw_frame = self.cap.read()

        if not ret:
            print("Failed to capture photo")
            return

        raw_frame = cv2.resize(raw_frame, (self.width, self.height))

        processed_frame, mask, find_ball, error = self.process_frame(raw_frame)
        print(f'find_ball: {find_ball}, error: {error}')
        cv2.imwrite(filename, processed_frame)
        cv2.imwrite(f"mask_{filename}", mask)

        print("finish")
        print("find_ball:", find_ball)
        print("error:", error)

    def close(self):
        if self.closed:
            return
        self.cap.release()
        cv2.destroyAllWindows()
        self.closed = True
        print("Camera and Windows closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with Camera() as tracker:
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            print("find_ball:", find_ball, "error:", error)
