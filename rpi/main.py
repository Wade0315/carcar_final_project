import time
import rpi.camera as camera
from enum import Enum

class Status(Enum):
    ERROR = 0
    NOT_FOUND = 1
    FOUND = 2
    CLOSE_ENOUGH = 3
    OUT_OF_BOUND = 4
    IDLE = 5


def main():
    state = Status.IDLE
    with camera.Camera() as cam: #open camera
        state = Status.NOT_FOUND
        for ball_detected in cam.streaming():
            if state == Status.IDLE:
                if ball_detected:
                    state = Status.FOUND


if __name__ == "__main__":
    main()