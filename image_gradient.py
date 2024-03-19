"""
Calculate image gradients

Contributors: 
    Thijs Willems
    
KU Leuven, LMSD
"""

import cv2 as cv
import matplotlib.pylab as plt
import numpy as np
import scipy.signal
from math import floor, ceil

def calc_gradient(image, kernel='central_fd', prefilter_gauss=True, verbosityFlag=0):
    """
    Computes gradient of input image, using the specified convoluton kernels.

    :param image: image to compute gradient of (2d numpy array).
    :type image: ndarray
    :param kernel: specifies the gradient calculation kernel type, specified as a string indicating the type or as a
    list of ndarrays that contains the kernels for the x and y direction. String types are:
        'central_fd': central finite difference kernel.
    :type kernel: str, [ndarray, ndarray]
    :param prefilter_gauss: bool to determine is a gaussian prefilter is used or not
    :type prefilter_gauss: bool

    :return: [gx, gy] (numpy array): gradient images with respect to x and y direction.
    :rtype: ndarray
    """
    if kernel == 'cd':
        # Central differences
        x_kernel = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]])/2
        y_kernel = np.transpose(x_kernel)
    elif kernel == 'sobel':
        # Sobel gradient
        x_kernel = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])/8
        y_kernel = np.transpose(x_kernel)
    elif 'sharr':
        # Sharr gradient
        x_kernel = np.array([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]])/32
        y_kernel = np.transpose(x_kernel)
    elif 'DoG':
        # Derivatives of Gaussian
        #TODO
        pass
        # x, y = np.mechgrid(floor(-3*sigma):ceil(3*sigma),floor(-3*sigma):ceil(3*sigma))
        # x_kernel = -(y./(2*pi*sigma^4)).*exp(-(x.^2+y.^2)/(2*sigma^2))
        # y_kernel = -(x./(2*pi*sigma^4)).*exp(-(x.^2+y.^2)/(2*sigma^2))
    elif not isinstance(kernel, str) and len(kernel) == 2 and kernel[0].shape[1] >= 3 and kernel[1].shape[0] >= 3:
        x_kernel = kernel[0]
        y_kernel = kernel[1]
    else:
        raise ValueError(
            'Please input valid gradient convolution kernels!')

    g_x = scipy.signal.convolve2d(image.astype(float), x_kernel, mode='same')
    g_y = scipy.signal.convolve2d(image.astype(float), y_kernel, mode='same')
    
    # Sobel derivatives
    # gradx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    # grady = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    # Scharr derivatives
    # gradx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=cv.FILTER_SCHARR)
    # grady = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=cv.FILTER_SCHARR)

    return np.array([g_x, g_y], dtype=np.float64)

def calc_gradient_fast(image):
    """
    Fast gradient computation (@author: pyidi package).

    Compute the gradient of image in both directions using central
    difference weights over 3 points.

    !!! WARNING:
    The edges are excluded from the analysis and the returned image
    is smaller then original.

    :param image: image to compute gradient of (2d numpy array)
    :type image: ndarray

    :return: gradient in x and y direction
    :rtype: ndarray, ndarray
    """
    im1 = image[2:]
    im2 = image[:-2]
    gy = (im1 - im2) / 2

    im1 = image[:, 2:]
    im2 = image[:, :-2]
    gx = (im1 - im2) / 2

    return gx[1:-1], gy[:, 1:-1]
