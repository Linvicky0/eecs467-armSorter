import os
import cv2
import numpy as np
import glob
import argparse


parser = argparse.ArgumentParser(description="Resize images")
parser.add_argument("--size", type=int, default=640, help="Resize size")
parser.add_argument("--input_dir", type=str, required=True, help="directory of raw images")
args = parser.parse_args()

print("Size:", args.size) # 640 by default

# input_dir = "data/images"
input_dir = args.input_dir
output_dir = "resized_images"

os.makedirs(output_dir, exist_ok=True)


def letterbox(image, new_size=640):
    h, w = image.shape[:2]

    scale = min(new_size / w, new_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # create padded image
    canvas = 255 * np.ones((new_size, new_size, 3), dtype=np.uint8)

    # center the image
    x_offset = (new_size - new_w) // 2
    y_offset = (new_size - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

files = glob.glob(f"{input_dir}/*.png", recursive=True)
for path in files:
    img = cv2.imread(path)
    resized = letterbox(img, args.size)
    filename = path.split('/')[-1]
    cv2.imwrite(os.path.join(f"{output_dir}", filename), resized)


