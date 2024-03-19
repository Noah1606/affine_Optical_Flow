# Fuctions related to data conversion:
#   - Change image color scheme or bit denpth
#
# Contributors:
#   Thijs Willems
#
# KU Leuven, LMSD

import numpy as np
import cv2 as cv
import re

def change_color_scheme(image, old_color_scheme, new_color_scheme):
    """change_color_scheme Change the color scheme of an image

    Parameters
    ----------
    image : numpy.ndarray
        Input image. Must be 8-bit unsigned, 16-bit unsigned, or single-precision floating-point. 
    old_color_scheme : _type_
        _description_
    new_color_scheme : _type_
        _description_
    """    
    
    if old_color_scheme == 'BGR' and new_color_scheme == 'GRAY':
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif old_color_scheme == 'GRAY' and new_color_scheme == 'BGR':
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif old_color_scheme == 'GRAY' and new_color_scheme == 'RGB':
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
    return image
    
def change_bit_depth(image, new_bit_depth, plot='off'):
    
    nr_bit_depth = [int(s) for s in re.findall(r'\d+', new_bit_depth)][-1]
        
    max_bit = 2**nr_bit_depth - 1
    
    image_temp = image.astype('float64')
    image_temp = image - image.min()
    image_temp = image_temp / image_temp.max()
    image_temp = image_temp * max_bit

    res_image = image_temp.astype(new_bit_depth)

    if plot == 'on':
        plt.imshow(res_image)
        plt.show()
    
    return res_image

