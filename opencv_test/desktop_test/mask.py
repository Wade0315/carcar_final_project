import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 40, 255])

try :
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        roi_frame = frame[0:1920, 0:1080]
        
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        kernel_close = np.ones((5,5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)

        cv2.imshow('1. Raw Mask', white_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("end")
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
