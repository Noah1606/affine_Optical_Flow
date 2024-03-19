# Image pyramids
#
# Contributors:
#   Thijs Willems
#
# KU Leuven, LMSD

import cv2 as cv

def image_pyramid(image, pyr_level, pyr_method, pyr_sigma, verbosityFlag=0):
    # Image pyramid
    img_tmp = image
    
    img_seq = []
    img_seq.append(img_tmp)
    
    for i_pyr in range(pyr_level):
        if pyr_method == 'OpenCV':
            img_tmp = cv.pyrDown(img_tmp)
        img_seq.append(img_tmp)

        if verbosityFlag >= 2:
            pass
    return img_seq