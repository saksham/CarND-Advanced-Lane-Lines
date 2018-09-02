#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from collections import namedtuple

import cv2
import numpy as np

from src import utils
from src.utils import logger, CHESS_BOARD_PATTERN_SIZE, CALIBRATION_IMAGES_LOCATION

CalibrationImage = namedtuple('CalibrationImage', ['filename', 'gray', 'img_pts', 'obj_pts'])


class Camera(object):
    def __init__(self):
        self._mtx = None
        self._mtx_inv = None
        self._dist = None

    def calibrate(self, display=True):
        nx, ny = CHESS_BOARD_PATTERN_SIZE
        calibration_images = []

        for f in glob.glob(CALIBRATION_IMAGES_LOCATION):
            img = cv2.imread(f)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESS_BOARD_PATTERN_SIZE)

            if not ret:
                logger.warn('Not all corners were found in %s, skipping...', f)
                continue
            objp = np.zeros((nx * ny, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
            calibration_images.append(CalibrationImage(f, gray, corners, objp))

            if display:
                window = cv2.namedWindow('ChessBoard')
                corners_img = cv2.drawChessboardCorners(img, CHESS_BOARD_PATTERN_SIZE, corners, True)
                cv2.imshow(window, corners_img)
                cv2.waitKey(250)

        obj_pts = [ci.obj_pts for ci in calibration_images]
        img_pts = [ci.img_pts for ci in calibration_images]
        im_shape = calibration_images[0].gray.shape

        ret, self._mtx, self._dist, _, ___ = cv2.calibrateCamera(obj_pts, img_pts, im_shape[::-1], None, None)
        self._mtx_inv = np.linalg.inv(self._mtx)

    def undistort(self, img):
        return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)

    def find_chess_board_perspective_transform_pts(self, img, offset=100):
        undistorted = self.undistort(img)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESS_BOARD_PATTERN_SIZE, None)
        if not ret:
            raise Exception('Corners not found')

        img_size = (gray.shape[1], gray.shape[0])

        nx, ny = CHESS_BOARD_PATTERN_SIZE
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        dst = np.float32(
            [[offset, offset], [img_size[0] - offset, offset], [img_size[0] - offset, img_size[1] - offset],
             [offset, img_size[1] - offset]])

        return src, dst

    def warp_image(self, img, perspective_transform_matrix):
        undistorted = self.undistort(img)
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(undistorted, perspective_transform_matrix, img_size, flags=cv2.INTER_LINEAR)

    def perform_perspective_transform(self, img, src, dst):
        src_arr = utils.to_array(src, np.float32)
        dst_arr = utils.to_array(dst, np.float32)

        logger.debug('Src for perspective transform: %a', src_arr)
        logger.debug('Dest for perspective transform: %a', dst_arr)
        perspective_transform_matrix = cv2.getPerspectiveTransform(src_arr, dst_arr)

        undistorted = self.undistort(img)
        warped = self.warp_image(img, perspective_transform_matrix)
        cv2.polylines(undistorted, np.int_([src]), True, (255, 0, 0), thickness=4)
        cv2.polylines(warped, np.int_([dst]), True, (255, 0, 0), thickness=4)
        return undistorted, warped, perspective_transform_matrix
