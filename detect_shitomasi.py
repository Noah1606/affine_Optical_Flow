'''
Functions for Shi-Tomasi corner point detection.

Contributors:
    Thijs Willems
    
KU Leuven, LMSD
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from image_data import read_image

def detect_shitomasi(image, shitomasi_parameters=None, mask=None,
               subpixel_refinement='off', subpixel_parameters=None,
               iteration_criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 0.0001),
               processor='cpu',
               verboseFlag=0):
    """
    Determine good features to track in each input image based on a Shi-Tomasi feature detection.

    Parameters
    ----------
    image : numpy.ndarray
        Image to detect the ArUco markers in.
    shitomasi_parameters : dict(maxCorners=int, qualityLevel=float, minDistance=int, blockSize=int_odd) or None, default=None
        Parameters for the Shi-Tomasi method.
        If None: dict(maxCorners=1, qualityLevel=0.01, minDistance=15, blockSize=5) is used.
    mask : str or numpy.ndarray
        Binary image defining the area (mask) to seacrh for corners.
    subpixel_refinement : str, default='off'
        Determine if the subpixel refinement is used. Options: 'on'/'off'.
    subpixel_parameters : dict(winSize=(int_odd, int_odd), zeroZone=(int, int)), default=None
        Dictionary with the subpixel refinement parameters 'winSize' and 'zeroZone'.
        If None, dict(winSize=(11, 11), zeroZone=(-1, -1)) is used.
    iteration_criteria : tuple, default=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 0.0001)
        Iteration criteria for the subpixel refinement.
    processor : str, default='cpu'
        Processor used for the corner detection. Options: 'cpu'/'gpu'.
    verboseFlag : int, default=0
        Determine if the plots are shown or not.

    Returns
    -------
    features : numpy.ndarray
        A numpy.ndarray of shape=(# points, 1, 2=(u, v)) of the detected feature points.
        
    """
    # Parameters for Shi-Tomasi corner detection
    if shitomasi_parameters is None:
        feature_params = dict(maxCorners=1, qualityLevel=0.01, minDistance=15, blockSize=5)
    else:
        feature_params = shitomasi_parameters
        
    corners = np.zeros((1, 1, 2))

    if processor == 'cpu':
        p = cv.goodFeaturesToTrack(image, mask=read_image(mask),**feature_params)  # p: ndarray, shape = (# points, 1, 2(u,v))
    elif processor == 'gpu':
        gray_gpu = cv.cuda_GpuMat()
        gray_gpu.upload(image)

        good_corners = cv.cuda.createGoodFeaturesToTrackDetector(
            srcType=cv.CV_8UC1,
            maxCorners=feature_params['maxCorners'],
            qualityLevel=feature_params['qualityLevel'],
            minDistance=feature_params['minDistance'],
            blockSize=feature_params['blockSize'],
            useHarrisDetector=False,
            harrisK=0.04)
        p_gpu = good_corners.detect(gray_gpu)  # p_gpu: shape = (1, # points, 2(u,v))
        p = p_gpu.download().reshape(-1, 1, 2)

    if subpixel_refinement == 'on':
        if subpixel_parameters is None:
            subpixel_parameters = dict(winSize=(11, 11), zeroZone=(-1, -1))
        cv.cornerSubPix(image, p,
                        subpixel_parameters['winSize'], subpixel_parameters['zeroZone'],
                        iteration_criteria)  # result is written in p (same format)

    for corner_nr in range(p.shape[0]):
        corners = np.append(corners, [[[p[corner_nr, 0, 0], p[corner_nr, 0, 1]]]], axis=0)

    features = np.delete(corners, 0, 0)

    if verboseFlag >= 2:
        plot_shitomasi(image, features, display=True)
    return features

def plot_shitomasi(image, data, display=False):
    vis = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    for u, v in np.float32(data).reshape(-1, 2):
        vis[int(v), int(u), :] = [0, 255, 0]
    fig = plt.figure()
    plt.title('Shi-Tomasi feature points')
    plt.imshow(vis)
    plt.show()

    return vis

if __name__ == "__main__":
    pass

