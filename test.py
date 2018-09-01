#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import image as mpimg
from matplotlib import pyplot as plt

im = mpimg.imread('straight_lines1.jpg')
plt.imshow(im)
plt.show()
