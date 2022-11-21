import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt 
import math
import imUtils

folder = 'test_images/'
ori_img = imUtils.imread(folder + '1.cr3')

patchPos = imUtils.get4ColourPatchPos(ori_img.copy())

