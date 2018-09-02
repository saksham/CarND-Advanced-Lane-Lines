#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import image as mpimg
from matplotlib import pyplot as plt

from src import calibration


def calibrate():
    camera = calibration.Camera()
    camera.calibrate(False)
    return camera


def undistort_and_show(camera, image_path):
    im = mpimg.imread(image_path)
    undistorted = camera.undistort(im)
    plt.imshow(undistorted)
    plt.show()


if __name__ == "__main__":
    camera = calibrate()

    # In order to get image coordinates of lane line corners
    undistort_and_show(camera, "examples/straight_lines1.jpg")
