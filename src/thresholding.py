#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple

import cv2
import numpy as np

from src import utils

ThresholdParams = namedtuple('Thresholdparams', ['color_space', 'channel_index', 'ch_threshold', 'sx_threshold'])


class Thresholder(object):
    KERNEL_SIZE = 5

    def __init__(self, thresholds):
        self._thresholds = thresholds

    def threshold(self, img):
        binary = np.zeros(img.shape[:2], dtype=int)
        for threshold in self._thresholds:
            thresholded = Thresholder._apply_thresholds(img, threshold)
            binary |= thresholded
        return utils.normalise_pixel_values(binary)

    @staticmethod
    def _apply_thresholds(img, params):
        if params.color_space:
            converted = cv2.cvtColor(img, params.color_space)
        else:
            converted = img

        channel = converted[:, :, params.channel_index]

        # Sobel X
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=Thresholder.KERNEL_SIZE)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= params.sx_threshold[0]) & (scaled_sobel <= params.sx_threshold[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(channel)
        s_binary[(channel >= params.ch_threshold[0]) & (channel <= params.ch_threshold[1])] = 1

        # Stack each channel
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        return cv2.cvtColor(color_binary, cv2.COLOR_RGB2GRAY)
        # return s_binary
