from ultralytics import YOLO
import torch
import os
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # For Mac M1/M2/M3
else:
    device = "cpu"

DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "model_config.yml")


def train_model(model_path, output_filename):
    model = YOLO(model_path) # use yolov8s.pt to start from scratch or input a trained model output
    # Train the model
    train_results = model.train(
        workers=2,
        batch=8,
        data=CONFIG_PATH,
        epochs=30,  # number of training epochs
        imgsz=640,  # training image size
        device=device,
        iou=0.6,                    # IoU threshold for NMS (lower for dense clusters)
        conf=0.25,                  # confidence threshold
        augment=True,               # use data augmentation
        overlap_mask=True, 
    )

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

    export_path = f"{CURRENT_DIR}/{output_filename}"

    model.export(format="pt")
    os.rename(model_path, export_path)
    print(f"Saved to: {export_path}")


def run_model(model_path=
              os.path.abspath(os.path.join(CURRENT_DIR, "..", "best.pt")),
              image_path = f"{DATA_DIR}/val/img.png"):
    model = YOLO(model_path)
    # results[0].show()
    return model(image_path)


def eval_model(model_path=os.path.abspath(os.path.join(CURRENT_DIR, "..", "best.pt"))):
    raw_images_dir = glob.glob(f"{DATA_DIR}/images/")
    labels_dir = glob.glob(f"{DATA_DIR}/labels/")
    model = YOLO(model_path)
    return model.val()

def printHello():
    print("hello")