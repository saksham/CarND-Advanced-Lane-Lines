#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger('advanced-lane-lines')
FORMAT = '%(levelname)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logger.setLevel(LOG_LEVEL)


def plot_images(images_with_captions, fig_size=(24, 9)):
    cols = len(images_with_captions)
    rows = len(images_with_captions[0]) if type(images_with_captions[0]) == list else 1

    f, axes = plt.subplots(rows, cols, figsize=fig_size)

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


def to_array(list_or_array, dtype=np.int32):
    if type(list_or_array) == np.ndarray:
        return list_or_array

    return np.array(list_or_array, dtype=dtype)


def to_three_channels(single_channel):
    return np.dstack((single_channel, single_channel, single_channel))


def generate_line_pts(imshape, best_fit, factor=1):
    y = np.int_(np.linspace(0, imshape[0] - 1, imshape[0] // factor))
    x = np.int_(best_fit[0] * (y ** 2) + best_fit[1] * y + best_fit[2])
    return x, y


def normalise_pixel_values(img, factor=255.0):
    max_pixel = np.max(img) / factor
    return (img / max_pixel).astype(np.uint8)
