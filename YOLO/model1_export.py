from ultralytics import YOLO

model = YOLO(
    r"/home/waryt/carcar_final_project/YOLO/runs/detect/shuttle_yolov5nu_256_v2/weights/best.pt"
)

model.export(
    format="ncnn",
    imgsz=256,
)