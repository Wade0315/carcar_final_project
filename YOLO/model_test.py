from ultralytics import YOLO

model = YOLO(r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\runs\detect\shuttle_yolov8n_320\weights\best.pt")

model.predict(
    source=r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\stock\image_test_1",
    imgsz=256,
    conf=0.25,
    save=True
)