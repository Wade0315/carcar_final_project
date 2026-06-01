from ultralytics import YOLO

model = YOLO(r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\runs-v5n\detect\shuttle_yolov5n_256\weights\best.pt")
results = model.predict(
    source=r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\stock\image_test_1",
    imgsz=256,
    conf=0.25,
    save=True,
    project=r"C:\Users\waryt\Desktop\carcar_final_project\YOLO\runs-v5n\detect",
    name="predict"

)

