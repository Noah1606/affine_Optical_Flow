# Functions for image warping
#
# Contributors:
#   Thijs Willems
#
# KU Leuven, LMSD

import numpy as np

def warp_image(img, F, A, v, win_idx, win_size, verbosityFlag):
    # Warp image
    
    # Get the current window for the new frame
    xyq = A@win_idx + v
    xq = xyq[0, :]
    yq = xyq[1, :]
    
    # Image wrapping and interpolation of the new frame
    warped_image = F(yq,xq,grid=False)
    warped_image = warped_image.reshape([win_size[0]*2, win_size[1]*2])

    return warped_image
