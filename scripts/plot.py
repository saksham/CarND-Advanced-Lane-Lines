#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from matplotlib import image as mpimg
from matplotlib import pyplot as plt

# measured coordinates from lane corners for perspective transform
# img = mpimg.imread('examples/straight_lines1.jpg')

# measured lane marking for calculating curvature
img = mpimg.imread('output_images/test5.jpg.2.warped_orig.jpg')
print('Lane is approx {} metres'.format(math.sqrt((1122 - 1134) ** 2 + (668 - 557) ** 2)))

plt.imshow(img)
plt.show()
