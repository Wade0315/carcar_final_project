from ultralytics import YOLO

model = YOLO(
    r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\runs\detect\shuttle_yolov8n_320\weights\best.pt"
)

model.export(
    format="ncnn",
    imgsz=320
)