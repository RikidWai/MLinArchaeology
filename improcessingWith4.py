import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Read Image
path = os.path.join(os.getcwd(), 'test_images', '1.cr3')
ori_img = utils.imread(path)

utils.imshow(ori_img)
