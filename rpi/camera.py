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
        find_ball = False
        error = None
        candicate = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            self.contour_dealing(cnt, candicate)
        if len(candicate) > 0:
            find_ball = True
        error = self.choose_ball(candicate)
        return frame, mask, find_ball, error
    
    def choose_ball(self, candicate):
        error = min(candicate, key=lambda b: abs(b["error"]))
        return error

    def contour_dealing(self, frame, cnt, candicate):
            area = cv2.contourArea(cnt)
            if 800 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                (w, h) = rect[1]
                
                if w == 0 or h == 0: 
                    return
                
                ratio = min(w, h) / max(w, h)
                if ratio > 0.35:
                    find_ball = True
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                    
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    error_from_center = cx - self.width //2
                    cv2.putText(frame, "Ball", (cx-10, cy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    candicate.append({
                    "contour": cnt,
                    "rect": rect,
                    "area": area,
                    "cx": cx,
                    "cy": cy,
                    "error": error_from_center,
                    "ratio": ratio
                })

        

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
        processed_frame, mask, find_ball = self.process_frame(raw_frame)
        
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