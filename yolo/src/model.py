from ultralytics import YOLO
import os
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(CURRENT_DIR, "..", "data")
CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "model_config.yml")



def run_model(model_path=
              os.path.abspath(os.path.join(CURRENT_DIR, "..", "best.pt")),
              image_path = f"{DATA_DIR}/val/img.png"):
    model = YOLO(model_path)
    # results[0].show()
    return model(image_path)
