from ultralytics import YOLO

model = YOLO("yolov5nu.pt")

model.train(
    data="shuttle.v5i.yolov5pytorch/data.yaml",
    imgsz=256,
    epochs=100,
    batch=16,
    device=0,
    workers=0,
    patience=30,
    cache=True,
    name="shuttle_yolov5nu_256_v2"
)