from ultralytics import YOLO

model = YOLO(
    r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\runs-v5n\detect\shuttle_yolov5n_256\weights\best.pt"
)

model.export(
    format="ncnn",
    imgsz=256,
)