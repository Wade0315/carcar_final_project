import cv2
import time
import logging
import os
from picamera2 import Picamera2
from camera_base import CameraBase

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


class Camera(CameraBase):
    def __init__(self, width=320, height=240, flip_code=-1):
        super().__init__(width, height, flip_code)

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
                raw_frame = self.picam2.capture_array()
                raw_frame = self.fix_orientation(raw_frame)
                if at_frame % 3 == 0:
                    processed_frame, floor_mask, badminton_mask, find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error            
                at_frame += 1
                if processed_frame is not None:
                    cv2.imshow('Robot View', processed_frame)
                if badminton_mask is not None and floor_mask is not None:
                    display_mask = self.build_display_mask(floor_mask, badminton_mask)
                    cv2.imshow('White Mask', display_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.close()
        
    def single_test(self, filename="test_capture.jpg"):
        logger.info("capturing photo...")
        raw_frame = self.picam2.capture_array()
        raw_frame = self.fix_orientation(raw_frame)
        processed_frame, floor_mask, badminton_mask, find_ball, error = self.process_frame(raw_frame)
        logger.info("find_ball=%s error=%s", find_ball, error)
        cv2.imwrite(f"/home/waryt/Desktop/{filename}", processed_frame)
        display_mask = self.build_display_mask(floor_mask, badminton_mask)
        cv2.imwrite(f"/home/waryt/Desktop/mask_{filename}", display_mask)
        logger.info("finish")
    
    def close(self):
        if self.closed:
            return
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.closed = True
        logger.info("Camera and Windows closed.")

if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        tracker.single_test()        
        #tracker.streaming()
