from ultralytics import YOLO
import os
import glob
import cv2
import numpy as np


def load_model():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "best.pt"))
    return YOLO(model_path)


def resizeImage(image, new_size=640):
    h, w = image.shape[:2]

    scale = min(new_size / w, new_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # blank white canvas (640x640)
    canvas = 255 * np.ones((new_size, new_size, 3), dtype=np.uint8)

    # use offsets to leave the sides white if new_w != new_size or new_h != new_size
    x_offset = (new_size - new_w) // 2
    y_offset = (new_size - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def get_bbox_pixels(bbox, original_shape, new_size=640):
    """
    bbox: (x1, y1, x2, y2) from the resized + padded image fed to model
    original_shape: (h, w) from camera from
    """
    orig_h, orig_w = original_shape[:2]

    scale = min(new_size / orig_w, new_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale) # closest size to new_size while maintaining object shapes in image

    x_offset = (new_size - new_w) // 2
    y_offset = (new_size - new_h) // 2

    x1, y1, x2, y2 = bbox

    # remove padding if there is any
    x1 -= x_offset
    x2 -= x_offset
    y1 -= y_offset
    y2 -= y_offset

    # undo cv2.resize
    x1 /= scale
    x2 /= scale
    y1 /= scale
    y2 /= scale

    return int(x1), int(y1), int(x2), int(y2)

def find_target_blocks(model, image, target_blocks):
    resized = resizeImage(image)
    result = model(resized)[0]
    found_blocks = {}
    class_names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]

        if class_name in target_blocks:
            confidence = round(box.conf[0].item(), 2)
            bbox = box.xyxy[0].tolist()
            bbox = get_bbox_pixels(bbox, image.shape)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            center = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if class_name in found_blocks:
                found_blocks[class_name].append({"confidence": confidence, "center": center, "label": class_name})
            else:
                found_blocks[class_name] = [{"confidence": confidence, "center": center, "label": class_name}]
    return found_blocks