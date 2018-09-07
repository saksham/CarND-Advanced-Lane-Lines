#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

from src import thresholding

_CONFIG = {
    "default": {
        "image-shape": (720, 1280, 3),
        "sliding-window": {
            "n-windows": 10,
            "margin": 100,
            "min-pix": 50
        },
        "polygon-window": {
            "margin": 100
        }
    },
    "project-video": {
        "thresholds": [thresholding.ThresholdParams(cv2.COLOR_RGB2HLS, 2, (200, 255), (15, 50)),
                       thresholding.ThresholdParams(None, 0, (215, 255), (15, 50))],
    },
    "challenge-video": {
        "thresholds": [thresholding.ThresholdParams(cv2.COLOR_RGB2HLS, 2, (100, 255), (15, 50)),
                       thresholding.ThresholdParams(None, 0, (215, 255), (15, 50))],
        "sliding-window": {
            "n-windows": 10,
            "margin": 50,
            "min-pix": 50
        },
        "polygon-window": {
            "margin": 50
        }
    },
    "harder-challenge-video": {
        "thresholds": [thresholding.ThresholdParams(cv2.COLOR_RGB2HLS, 2, (100, 255), (15, 50)),
                       thresholding.ThresholdParams(None, 0, (215, 255), (15, 50))],
    }
}


def get_config(config_name):
    return {**_CONFIG["default"], **_CONFIG[config_name]}
