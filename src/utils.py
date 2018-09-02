#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
from matplotlib import pyplot as plt

# Constants
CALIBRATION_IMAGES_LOCATION = "camera_cal/*.jpg"
CHESS_BOARD_PATTERN_SIZE = (9, 6)

logger = logging.getLogger('advanced-lane-lines')
FORMAT = '%(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logger.setLevel(LOG_LEVEL)


def plot_images(images_with_captions):
    cols = len(images_with_captions)
    rows = len(images_with_captions[0]) if type(images_with_captions[0]) == list else 1

    f, axes = plt.subplots(rows, cols, figsize=(24, 9))

    f.tight_layout()
    for i in range(cols):
        data = []
        if type(images_with_captions[i]) == list:
            for j in range(rows):
                img, caption = images_with_captions[i][j]
                if len(axes.shape) == 1:
                    ax = axes[j]
                else:
                    ax = axes[j, i]
                data.append([img, caption, ax, 10])
        else:
            img, caption = images_with_captions[i]
            ax = axes[i]
            data.append([img, caption, ax, 30])

        for d in data:
            img, caption, ax, font_size = d
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(caption, fontsize=font_size)

    plt.subplots_adjust(left=0., right=1, top=1, bottom=0.)


def rectangle_pts_inside_image(img, x_offset=200, y_offset=0):
    height, width, _ = img.shape
    return [
        [x_offset, height - y_offset], [width - x_offset, height - y_offset],
        [width - x_offset, y_offset], [x_offset, y_offset]
    ]


def bgr2rgb(image):
    return image[:, :, ::-1]


def to_array(list_or_array, dtype=np.int32):
    if type(list_or_array) == np.ndarray:
        return list_or_array

    return np.array(list_or_array, dtype=dtype)
