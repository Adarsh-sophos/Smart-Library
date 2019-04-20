import cv2
import sys
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import main

import spine_segmentation as seg

def test_spine_seg():
    path = main.BASE_PATH + "/data/query_images"

    image_name = [file_name for file_name in os.listdir(path)]

    #print(file_name)

    for image in image_name:
        img_path = path + '/' + image
        seg.get_book_lines(img_path, debug = False)


test_spine_seg()
