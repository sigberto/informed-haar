import cv2
import os


def avg_edge_map(dir):
    image_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        image_filenames.extend(filenames)
        break
    avg_edge_map = None
    for image_filename in image_filenames:
        img = cv2.imread(image_filename)
