import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt 
import math

def imread(path, percent = 50):
  _ , extension  = os.path.splitext(path)
  if 'cr'  in extension or 'CR' in extension:
    raw = rawpy.imread(path) # access to the RAW image
    rgb = raw.postprocess() # a numpy RGB array
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # the OpenCV image

    scale_percent = percent # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)   
    img = cv2.resize(img, dim)
  else: 
    img = cv2.imread(path)

  return img 

def imshow(img): 
  if img.shape[1] >= 5000 and img.shape[1] >= img.shape[0] :
    width = 1000
    height = int(img.shape[0] * 1000 / img.shape[1])
    dim = (width, height)
    # # resize image
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = cv2.resize(img, dim)   
  elif img.shape[0] >= 5000 and img.shape[0] >= img.shape[1]:
    width = int(img.shape[1] * 1000 / img.shape[0])
    height = 1000
    dim = (width, height)   
    # # resize image
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = cv2.resize(img, dim)
  cv2.imshow(img)

def whiteBalance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def contrastStretching(img):
  norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  norm_img = (255*norm_img).astype(np.uint8)
  return norm_img
