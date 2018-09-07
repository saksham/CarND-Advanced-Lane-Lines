#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from matplotlib import image as mpimg
from matplotlib import pyplot as plt

# measured coordinates from lane corners for perspective transform
# img = mpimg.imread('examples/straight_lines1.jpg')

# measured lane marking for calculating curvature
img = mpimg.imread('output/images/test5.jpg.2.warped_orig.jpg')
print('Distances is approx {} and {} pixels in X and Y directions respectively'.format(1167.92 - 169.8, 673-554))



plt.imshow(img)
plt.show()
