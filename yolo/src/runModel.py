from ultralytics import YOLO
import torch
from datetime import datetime
import os

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # For Mac M1/M2/M3
else:
    device = "cpu"

DATA_DIR = "../data/"
IMAGE_ID = 0

# Load a model
pretrained_model = "yolov8s.pt"
model = YOLO(pretrained_model)

# Train the model
train_results = model.train(
    workers=2,
     batch=8,
    data="../model_config.yml",
    epochs=30,  # number of training epochs
    imgsz=640,  # training image size
    device=device,
    iou=0.6,                    # IoU threshold for NMS (lower for dense clusters)
    conf=0.25,                  # confidence threshold
    augment=True,               # use data augmentation
    overlap_mask=True, 
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model(f"{DATA_DIR}/images/val/frame_{str(IMAGE_ID)}.png")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_path = f"cubes_model_{timestamp}.pt"

model.export(format="pt")
os.rename(pretrained_model, export_path)
print(f"Saved to: {export_path}")
