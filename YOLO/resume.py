from ultralytics import YOLO

model = YOLO("../runs/detect/shuttle_yolov5nu_256_v2/weights/last.pt")
model.train(resume=True)
