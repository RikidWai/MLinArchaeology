import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import rawpy
import os


def imread(path, percent=50):
    """
    Read image in raw format or jpg

    Parameters
    ----------
    path : string
        the name of the file to read.

    percent : int
        the percentage to scale 

    Returns
    -------
    result : numpy array
        data stored in the file

    """
    _, extension = os.path.splitext(path)
    if 'cr' in extension or 'CR' in extension:
        raw = rawpy.imread(path)  # access to the RAW image
        rgb = raw.postprocess()  # a numpy RGB array
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # the OpenCV image

        scale_percent = percent  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim)
    else:
        img = cv2.imread(path)

    return img


def imshow(img):
    """
    Show image 

    Parameters
    ----------
    img : numpy
        the image to show

    """
    
    
    
    # Resize if image is too large
    if img.shape[1] >= 1000 and img.shape[1] >= img.shape[0]:
        width = 1000
        height = int(img.shape[0] * 1000 / img.shape[1])
        dim = (width, height)
        img = cv2.resize(img, dim)
    elif img.shape[0] >= 1000 and img.shape[0] >= img.shape[1]:
        width = int(img.shape[1] * 1000 / img.shape[0])
        height = 1000
        dim = (width, height)
        img = cv2.resize(img, dim)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
