import cv2
import numpy as np

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