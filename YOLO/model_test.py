from ultralytics import YOLO

model = YOLO(r"/home/waryt/carcar_final_project/runs/detect/shuttle_yolov5nu_256_v2/weights/best.pt")
results = model.predict(
    source=r"/home/waryt/carcar_final_project/YOLO/shuttle.v5i.yolov5pytorch/valid/images",
    imgsz=256,
    conf=0.25,
    save=True,
    project=r"/home/waryt/carcar_final_project/runs/detect_v5nu_v2",
    name="predict"

)

