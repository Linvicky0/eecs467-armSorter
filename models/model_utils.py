import itertools

TEST_DATA_DIR = "test_data_v2"
TRAIN_DATA_DIR = "train_data_v2"
camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]
rainbow_hex = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF", "#800080"]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]
SHAPES = ["sphere", "cube"]
SIZES = ["small", "large"]
NUM_CLASSES = len(COLORS) * len(SHAPES) * len(SIZES)
combinations = list(itertools.product(SIZES, COLORS, SHAPES))
LABEL_MAP = {i + 1: combo for i, combo in enumerate(combinations)}


def get_size_from_id(id):
    return LABEL_MAP[id][0]

def get_color_from_id(id):
    return LABEL_MAP[id][1]

def get_shape_from_id(id):
    return LABEL_MAP[id][2]