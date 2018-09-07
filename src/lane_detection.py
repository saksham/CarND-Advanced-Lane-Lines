#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
from collections import namedtuple

import cv2
import numpy as np

from src import utils, thresholding, config

logger = utils.logger

InterimImages = namedtuple('InterimImages', ['warped', 'thresholded', 'search_window', 'polynomial'])

Y_M_PER_PIX = 30 / 720
X_M_PER_PIX = 3.7 / 700


class Lane(object):
    NUM_OF_FRAMES_TO_KEEP = 10
    VALID_CURVATURE_LIMITS = (10, 100000)

    def __init__(self, name, img_shape):
        self._name = name
        self._img_shape = img_shape
        self._last_median_curvature = None
        self._curvatures = queue.deque(maxlen=Lane.NUM_OF_FRAMES_TO_KEEP)
        self._last_median_best_fit = None
        self._best_fit_window = queue.deque(maxlen=Lane.NUM_OF_FRAMES_TO_KEEP)

    def validate(self, curvature):
        if curvature < Lane.VALID_CURVATURE_LIMITS[0] or curvature > Lane.VALID_CURVATURE_LIMITS[1]:
            return False

        if self._last_median_curvature is not None:
            if curvature > 10 * self._last_median_curvature or curvature < 0.001 * self._last_median_curvature:
                return False

        return True

    @staticmethod
    def calculate_curvature(x, y, y_eval):
        # Fit new polynomials to x,y in world coordinates
        fit = np.polyfit(y * Y_M_PER_PIX, x * X_M_PER_PIX, 2)
        return ((1 + (2 * fit[0] * y_eval * Y_M_PER_PIX + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    def update(self, x, y):
        best_fit = np.polyfit(y, x, 2)
        curvature = self.calculate_curvature(x, y, self._img_shape[1])

        if not self.validate(curvature):
            logger.warn('Invalid curvature found.')
            if len(self._best_fit_window) > 0:
                self._best_fit_window.pop()
                self._curvatures.pop()
        else:
            self._best_fit_window.append(best_fit)
            self._last_median_best_fit = np.median(self._best_fit_window, 0)
            self._curvatures.append(curvature)

    @property
    def best_fit_px(self):
        if not self.has_best_fit:
            return None
        return self._last_median_best_fit

    @property
    def has_best_fit(self):
        return len(self._best_fit_window) > 0

    @property
    def curvature_m(self):
        x_pix, y_pix = utils.generate_line_pts(self._img_shape, self.best_fit_px, 2)
        return Lane.calculate_curvature(x_pix, y_pix, self._img_shape[0])


class LaneDetector(object):
    def __init__(self, camera, perspective_transform_mat, detector_name):
        self._camera = camera
        self._perspective_transform_mat = perspective_transform_mat
        self._perspective_transform_mat_inv = np.linalg.inv(perspective_transform_mat)
        cfg = config.get_config(detector_name)
        self._thresholder = thresholding.Thresholder(cfg["thresholds"])
        self._img_shape = cfg["image-shape"]
        self._sliding_window = cfg["sliding-window"]
        self._polygon_window = cfg["polygon-window"]
        self._left_lane = Lane('left', self._img_shape)
        self._right_lane = Lane('right', self._img_shape)

    def find_lane_pixels(self, binary_warped):
        if self._left_lane.has_best_fit and self._right_lane.has_best_fit:
            return self._find_lane_px_using_poly(binary_warped)
        return self._find_lane_px_using_sliding_windows(binary_warped)

    def _find_lane_px_using_sliding_windows(self, binary_warped):
        # Take a histogram of the bottom 3/4 of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 4:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        nwindows = self._sliding_window["n-windows"]
        margin = self._sliding_window["margin"]
        minpix = self._sliding_window["min-pix"]

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzeroy, nonzerox = binary_warped.nonzero()

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        out_img = utils.to_three_channels(utils.normalise_pixel_values(binary_warped))

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If we found > minpix pixels, recenter next window
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        return leftx, lefty, rightx, righty, out_img

    def _find_lane_px_using_poly(self, binary_warped):
        # HYPERPARAMETER
        margin = self._polygon_window["margin"]

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Get the last best fits
        left_fit = self._left_lane.best_fit_px
        right_fit = self._right_lane.best_fit_px

        # Filter out pixels within certain margin from the best fit
        left_fit_x = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
        left_lane_inds = ((nonzerox > (left_fit_x - margin)) & (nonzerox < (left_fit_x + margin)))
        right_fit_x = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]
        right_lane_inds = ((nonzerox > (right_fit_x - margin)) & (nonzerox < (right_fit_x + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img = utils.to_three_channels(utils.normalise_pixel_values(binary_warped))

        left_fit = self._left_lane.best_fit_px
        right_fit = self._right_lane.best_fit_px
        left_fit_x, left_fit_y = utils.generate_line_pts(binary_warped.shape, left_fit, 2)
        right_fit_x, right_fit_y = utils.generate_line_pts(binary_warped.shape, right_fit, 2)

        window_img = np.zeros_like(out_img)

        # # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # # Generate a polygon to illustrate the search window area
        # # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, left_fit_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, left_fit_y])))])

        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, right_fit_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, right_fit_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return leftx, lefty, rightx, righty, out_img

    def draw_final(self, undistorted, offset):
        img_w, img_h = undistorted.shape[1], undistorted.shape[0]
        color_warp = np.zeros_like(undistorted).astype(np.uint8)

        left_x, left_y = utils.generate_line_pts(undistorted.shape, self._left_lane.best_fit_px, 2)
        right_x, right_y = utils.generate_line_pts(undistorted.shape, self._right_lane.best_fit_px, 2)

        pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))], dtype=np.int32)
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))], dtype=np.int32)
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, pts, (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = cv2.warpPerspective(color_warp, self._perspective_transform_mat_inv, (img_w, img_h))
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, new_warp, 0.3, 0)

        cv2.putText(result, "L. Curvature: %.2f km" % (self._left_lane.curvature_m / 1000), (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "R. Curvature: %.2f km" % (self._right_lane.curvature_m / 1000), (50, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "C. Position: %.2f m" % offset, (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        return result

    def draw_polynomial(self, binary_warped):
        out_img = np.zeros_like(binary_warped).astype(np.uint8)
        out_img = utils.to_three_channels(out_img)

        left_fit = self._left_lane.best_fit_px
        right_fit = self._right_lane.best_fit_px
        left_x, left_y = utils.generate_line_pts(binary_warped.shape, left_fit, 2)
        right_x, right_y = utils.generate_line_pts(binary_warped.shape, right_fit, 2)

        pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
        pts = [np.int_(pts_left), np.int_(pts_right)]
        out_img = cv2.polylines(out_img, pts, False, [255, 255, 0], thickness=10)

        return out_img

    def calculate_offset(self):
        left_fit = self._left_lane.best_fit_px
        right_fit = self._right_lane.best_fit_px

        y_eval = self._img_shape[0]
        left_x = left_fit[0] * (y_eval ** 2) + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * (y_eval ** 2) + right_fit[1] * y_eval + right_fit[2]
        return (self._img_shape[1] - (right_x + left_x)) / 2 * X_M_PER_PIX

    def detect(self, img):
        # Apply thresholds
        im_thresholded = self._thresholder.threshold(img)

        # Warp the image
        im_warped = self._camera.warp_image(im_thresholded.astype(np.float), self._perspective_transform_mat)

        # Find the lane pixels
        left_x, left_y, right_x, right_y, im_search_window = self.find_lane_pixels(im_warped)

        # Update the lanes with pixels from the new image
        self._left_lane.update(left_x, left_y)
        self._right_lane.update(right_x, right_y)
        im_polynomial = self.draw_polynomial(im_warped)

        # Get the curvatures and offset
        offset = self.calculate_offset()
        curvature = (self._left_lane.curvature_m, self._right_lane.curvature_m)

        # Overlay lanes
        undistorted = self._camera.undistort(img)
        im_result = self.draw_final(undistorted, offset)

        interim_images = InterimImages(im_thresholded, im_warped, im_search_window, im_polynomial)
        return curvature, offset, interim_images, im_result
