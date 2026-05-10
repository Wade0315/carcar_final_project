import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 40, 255])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        roi_frame = frame[0:1920, 0:1080]
        
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        cv2.imshow('1. Raw Mask', white_mask)
        
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        kernel_close = np.ones((5,5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(area)
            if 2000 < area < 80000:
                
                rect = cv2.minAreaRect(cnt)
                (w, h) = rect[1]
                
                if w == 0 or h == 0:
                    continue
                    
                short_side = min(w, h)
                long_side = max(w, h)
                
                ratio = short_side / long_side
                
                if ratio > 0.35:
                    
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(roi_frame, [box], 0, (255, 0, 0), 2)
                    
                    center_x, center_y = int(rect[0][0]), int(rect[0][1])
                    cv2.putText(roi_frame, "Ball", (center_x-10, center_y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Pi 3B+ Robot View', roi_frame)
        # cv2.imshow('White Mask', white_mask) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("手動結束程式")
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()