import cv2
import numpy as np
import time
from picamera2 import Picamera2

class Camera:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 40, 255])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)

        self.target_x = None
        self.target_y = None
        self.last_error = None
        self.lost_count = 0
        self.max_lost_frames = 5
        self.max_tracking_distance = 50

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"AwbMode": 0})
        
        print("camera activating...")
        time.sleep(2)

    def process_frame(self, frame):
        #deal with single frame        
        candidate = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            self.contour_dealing(frame, cnt, candidate)
        
        find_ball, error, target = self.choose_ball(candidate)

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

            if self.lost_count <= self.max_lost_frames:
                return True, self.last_error, None
            else:
                self.target_x = None
                self.target_y = None
                self.last_error = None
                return False, None, None
    
    def contour_dealing(self, frame, cnt, candidate):
            area = cv2.contourArea(cnt)
            if 800 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                (w, h) = rect[1]
                
                if w == 0 or h == 0: 
                    return
                
                ratio = min(w, h) / max(w, h)
                if ratio > 0.35:
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                    
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    error_from_center = cx - self.width //2
                    cv2.putText(frame, "Ball", (cx-10, cy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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

        cv2.putText(
            frame,
            "TARGET",
            (cx - 25, cy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.line(
            frame,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            (0, 255, 255),
            1
        )

        cv2.line(
            frame,
            (self.width // 2, cy),
            (cx, cy),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"error={target['error']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    def streaming(self):
        print("Starting tracking... Press 'q' to quit.")
        at_frame = 0
        try:
            while True:
                raw_frame = self.picam2.capture_array()
                if at_frame % 3 == 0:
                    processed_frame, mask, find_ball, error = self.process_frame(raw_frame)
                yield find_ball, error            
                at_frame += 1
                cv2.imshow('Robot View', processed_frame)
                cv2.imshow('White Mask', mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.close()
        
    def single_test(self, filename="test_capture.jpg"):
        print("capturing photo...")
        raw_frame = self.picam2.capture_array()
        processed_frame, mask, find_ball, error = self.process_frame(raw_frame)
        print(f'find_ball: {find_ball}, error: {error}')
        cv2.imwrite(f"/home/waryt/Desktop/{filename}", processed_frame)
        cv2.imwrite(f"/home/waryt/Desktop/mask_{filename}", mask)
        print(f"finish")
    
    def close(self):
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Camera and Windows closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    with Camera() as tracker:
        tracker.single_test()        
        #tracker.streaming()