import cv2
import numpy as np

image_path = 'test_image.jpg' 
frame = cv2.imread(image_path)

if frame is None:
    print(f"{image_path} not find")
    exit()

roi_frame = frame.copy() 

lower_white = np.array([0, 180, 0])
upper_white = np.array([180, 255, 40])

hls = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HLS)
white_mask = cv2.inRange(hls, lower_white, upper_white)
cv2.imshow('1.Pure White Mask', white_mask)

kernel_open = np.ones((3,3), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_open)
kernel_close = np.ones((10,10), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close)
cv2.imshow('2. White Mask', white_mask)

contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"--- 開始分析照片 {image_path} ---")
for cnt in contours:
    area = cv2.contourArea(cnt)
    
    print(f"area: {area}")
        
    if 2000 < area < 80000:
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        
        if w == 0 or h == 0:
            continue
            
        short_side = min(w, h)
        long_side = max(w, h)
        
        ratio = short_side / long_side
        
        if ratio > 0.35:
            cv2.drawContours(roi_frame, [cnt], -1, (0, 255, 0), 2)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(roi_frame, [box], 0, (255, 0, 0), 2)
            
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            cv2.putText(roi_frame, "Ball", (center_x-10, center_y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print("---------------------------")

cv2.imshow('3. Result', roi_frame)

cv2.waitKey(0) 
cv2.destroyAllWindows()