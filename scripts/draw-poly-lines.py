#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = np.zeros((768, 1024, 3), dtype=np.uint8)

points = np.array([[0, 600], [400, 650], [500, 700], [1000, 800]])
print(points.shape)
cv2.polylines(img, [points], False, (233, 255, 0), thickness=10)

cv2.namedWindow('example')
cv2.imshow('example', img)
cv2.waitKey()
cv2.destroyAllWindows()
