import pyvista as pv
import numpy as np
import random
import os
import shutil
import cv2
from model_utils import *
import glob

def restart_dataset(dir_names=["test_data", "train_data"]):
    for dir_name in dir_names:
        print(f"Restarting {dir_name}...")
        if os.path.exists(dir_name): # delete dirs
            shutil.rmtree(dir_name)
        # Recreate directories
        os.makedirs(f"{dir_name}/test_images")
        os.makedirs(f"{dir_name}/labels")
        os.makedirs(f"{dir_name}/segmentations")


def generate_rainbow_dataset(output_dir, num_images=1, start_idx=0):


    for img_idx in range(start_idx, start_idx + num_images):
        # Create a plotter (off_screen=True lets it run in the background)
        plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
        plotter.set_background("black")
        plotter.camera_position = camera_position
        task_name = f"task-{img_idx + 1}"

        num_blocks = int(random.uniform(2, 10))
        block_ids = {}

        for i in range(num_blocks):
            id = random.randint(1, max([i for i in LABEL_MAP.keys()]))
            if id in block_ids:
                block_ids[id] += 1
            else:
                block_ids[id] = 1
            # print(f"adding {id}")
        
        for label_id, count in block_ids.items(): # for each class label, create a segmented image
            object_plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
            object_plotter.set_background("black")
            object_plotter.camera_position = camera_position
            # print(label_id, get_color_from_id(label_id))
            # continue
            for _ in range(count):
                block_size = get_size_from_id(label_id)
                block_color = get_color_from_id(label_id)
                block_shape = get_shape_from_id(label_id)

                # place objects
                pos = (random.uniform(-5, 5), random.uniform(-5, 5), 0)
                size_param = 0.5 if block_size == "small" else 1.0
                if block_shape == "cube":
                    shape = pv.Cube(center=pos, x_length=size_param, y_length=size_param, z_length=size_param)
                elif block_shape == "sphere":
                    shape = pv.Sphere(center=pos, radius=size_param * 0.6)

                object_plotter.add_mesh(shape, color="white", lighting=False) # add to segmented image
                plotter.add_mesh(shape, color=block_color) # add to 'raw' image

            img = object_plotter.screenshot(return_img=True)
            # Convert to binary mask (0=background, 255=object)
            mask = (img.sum(axis=2) > 0).astype(np.uint8) * 255

            # Save mask of segmented image
            cv2.imwrite(f"{output_dir}/labels/{task_name}-{label_id}.png", mask)

        # This creates the "image_X.png" file for your training
        image_path = f"{output_dir}/test_images/image_{task_name}.png"
        plotter.show(screenshot=image_path)
        print(f"Saved: {image_path}")


def gen_tasks(task_indices, dir_name):
    for idx in task_indices:
        generate_rainbow_dataset(num_images=1, start_idx=idx-1, output_dir=dir_name)


def delete_tasks(task_indices, dir_name):
    for idx in task_indices: # delete the raw test image
        task_name = f"task-{idx}"
        image_path = f"{dir_name}/test_images/image_{task_name}.png"
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        else:
            print(f"Image not found: {image_path}")

        files = glob.glob(f"{dir_name}/labels/{task_name}-*")
        for f in files:
            os.remove(f)
            print(f"Deleted: {f}")


if __name__ == "__main__":
    restart_dataset(["testImages"])
    generate_rainbow_dataset(num_images=3, start_idx=0, output_dir="testImages")
    # generate_rainbow_dataset(num_images=50, start_idx=0, output_dir=TEST_DATA_DIR)
    # to_delete = [2, 5, 15]
    # delete_tasks(to_delete, TEST_DATA_DIR)
    # gen_tasks(to_delete, TEST_DATA_DIR)
    pass
