import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt
import math
import imUtils
import configure as cfg
import colour

folder = 'test_images/'
img = imUtils.imread(folder + '1.CR2', 100)
imUtils.imshow(img)
img = imUtils.whiteBalance(img)
edged = imUtils.getEdgedImg(img)
imUtils.imshow(edged)