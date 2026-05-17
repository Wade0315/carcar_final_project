import cv2
import numpy as np
import time


class Camera:
    def __init__(self, width=320, height=240, camera_index=0):
        self.width = width
        self.height = height
        self.camera_index = camera_index

        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 40, 255])
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((10, 10), np.uint8)

        self.cap = cv2.VideoCapture(self.camera_index)

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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if 800 < area < 10000:
                rect = cv2.minAreaRect(cnt)
                (w, h) = rect[1]

                if w == 0 or h == 0:
                    continue

                ratio = min(w, h) / max(w, h)

                if ratio > 0.35:
                    find_ball = True

                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    error = cx - self.width // 2
                    cv2.putText(
                        frame,
                        "Ball",
                        (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )

        return frame, mask, find_ball, error

    def streaming(self):
        print("Starting tracking... Press 'q' to quit.")

        try:
            while True:
                ret, raw_frame = self.cap.read()

                if not ret:
                    print("Failed to read frame")
                    break

                raw_frame = cv2.resize(raw_frame, (self.width, self.height))

                processed_frame, mask, find_ball, error = self.process_frame(raw_frame)

                yield find_ball, error

                cv2.imshow("Robot View", processed_frame)
                cv2.imshow("White Mask", mask)

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

        processed_frame, mask, find_ball = self.process_frame(raw_frame)

        cv2.imwrite(filename, processed_frame)
        cv2.imwrite(f"mask_{filename}", mask)

        print("finish")

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera and Windows closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with Camera() as tracker:
        # tracker.single_test()
        for find_ball in tracker.streaming():
            print("find_ball:", find_ball)