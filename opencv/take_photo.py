import cv2
import time

print("activate camera...")
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("connot open camera")
    exit()

print("openning...")
time.sleep(2)

for i in range(5):
    cap.read()

ret, frame = cap.read()

if ret:
    file_name = 'test_image.jpg'
    cv2.imwrite(file_name, frame)
    print(f"save image: {file_name}")
else:
    print("error! cannot save image")

cap.release()