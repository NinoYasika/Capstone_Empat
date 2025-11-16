# from google.colab import drive
# drive.mount('/content/drive')

!pip install ultralytics --upgrade

import ultralytics
ultralytics.checks()

!pip install ultralytics roboflow supervision

from roboflow import Roboflow

rf = Roboflow(api_key="wraQ5Z8NaxoKsxyt9TFm")
project = rf.workspace("ayu-asipq").project("calory")
dataset = project.version(1).download("yolov11")


# !ls /content


import torch
print(torch.cuda.is_available())



from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/content/Calory-1/data.yaml",   # ← path yang benar
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20,
    pretrained=True,
    device=0   # GPU aktif
)


# !find /content/runs/detect -name "best.pt"


!pip install -qU ultralytics supervision

import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO('/content/runs/detect/train/weights/best.pt')

image = cv2.imread('/content/drive/MyDrive/Capstone 4/test_food.jpg')

# Inference
results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results).with_nms()

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=0.8,
    text_thickness=2,
    text_padding=5
)

# ------------ FORMAT LABEL PER BARIS -------------
labels = []
for class_id in detections.class_id:
    raw_name = model.names[int(class_id)]  # contoh: "Nasi -129 kal per 100gr-"

    # Pisahkan 2 bagian (nama & kalori)
    parts = raw_name.split("-")
    food_name = parts[0].strip()                   # "Nasi"
    calorie_raw = parts[1].replace("kal per", "").replace("100gr", "").strip()

    # Format lebih rapi → "129 kal/100gr"
    calorie_clean = f"{calorie_raw} kal/100gr"

    # Gabungkan jadi 2 baris
    label = f"{food_name}\n{calorie_clean}"
    labels.append(label)
# --------------------------------------------------

# Draw annotation
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

sv.plot_image(annotated_image)

detections


detections
