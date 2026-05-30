from ultralytics import  YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="shuttle.v1i.yolov8/data.yaml",
    imgsz=320,
    epochs=100,
    batch=4,
    workers=0,
    name="shuttle_yolov8n_320"
)