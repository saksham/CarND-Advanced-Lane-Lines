#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from matplotlib import pyplot as plt

# Constants
CALIBRATION_IMAGES_LOCATION = "camera_cal/*.jpg"
CHESS_BOARD_PATTERN_SIZE = (9, 6)

logger = logging.getLogger('advanced-lane-lines')
FORMAT = '%(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logger.setLevel(LOG_LEVEL)


def display_two_images(img1, caption1, img2, caption2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if len(img1.shape) == 2:
        ax1.imshow(img1, cmap='gray')
    else:
        ax1.imshow(img1)
    ax1.set_title(caption1, fontsize=50)

    if len(img2.shape) == 2:
        ax2.imshow(img2, cmap='gray')
    else:
        ax2.imshow(img2)
    ax2.set_title(caption2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def bgr2rgb(image):
    return image[:, :, ::-1]
