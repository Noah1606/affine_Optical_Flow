# Affine optical flow
#
# Contributors:
#   Thijs Willems
#
# KU Leuven, LMSD

import numpy as np
from math import sin, cos
from scipy.interpolate import RectBivariateSpline

from image_pyramids import image_pyramid
from image_gradient import calc_gradient
from warp_image import warp_image

def calc_affine_of(ref_image, new_image, ref_points, initial_guess=None, settings=None):
    # OF tracker between two frames
    # -> Only valid for small displacements or if good initial guesses are provided!!!
        
    # Initialization
    num_points = ref_points.shape[0]
    points_status = np.zeros((num_points,1))
    output_v = np.zeros((num_points, 1, 2))
    output_A = np.zeros((2, 2, num_points))

    # Affine model reduction matrix
    H0, selection = model_reduction_matrix(settings['model_reduction'], settings['motion_model'])

    # Convert images to float type (no uint8!)
    ref_image = ref_image.astype('float64')
    new_image = new_image.astype('float64')

    # Generate pyramid images
    ref_seqs = image_pyramid(ref_image, settings['pyr_params'][0], settings['pyr_params'][1], settings['pyr_params'][2])
    new_seqs = image_pyramid(new_image, settings['pyr_params'][0], settings['pyr_params'][1], settings['pyr_params'][2])

    # Loop over all the points
    u_ref = ref_points[:, 0, 0]
    v_ref = ref_points[:, 0, 1]

    # Generate image interpolant
    x_arr = np.arange(new_image.shape[1])
    y_arr = np.arange(new_image.shape[0])
    F = RectBivariateSpline(
            x=y_arr,
            y=x_arr,
            z=new_image,
            kx=3,
            ky=3,
            s=0
        )

    # Generate window indices
    win_x_arr = np.arange(-settings['win_size'][1], settings['win_size'][1])
    win_y_arr = np.arange(-settings['win_size'][0], settings['win_size'][0])
    [win_x, win_y] = np.meshgrid(win_x_arr,win_y_arr)
    win_idx = np.array([win_x.flatten(), win_y.flatten()])

    for i in range(num_points):
        # Initialization
        u = u_ref[i]
        v = v_ref[i]
        uv = np.array([[u], [v]])

        # Global initial guess for A and v
        if initial_guess is None:
            v_g = uv
            A_g = np.identity(2)
        else:
            v_g = initial_guess['v'][i, :]
            A_g = initial_guess['A'][:, :, i]

        # Initialization of affine tracking
        v_tmp = v_g/(2**(settings['pyr_params'][0]+1))
        A_tmp = A_g
        
        # Loop over all the pyramid levels
        for L in range(settings['pyr_params'][0]+1, 0, -1):
            #TODO: use different window sizes for different pyramid levels
            ref_tmp = ref_seqs[L-1]; new_tmp = new_seqs[L-1]
            v_tmp = 2*v_tmp
            v_tmp, A_tmp, point_st = calc_d(ref_tmp, new_tmp, v_tmp, A_tmp, uv/(2**(L-1)), H0, selection, F, win_idx, points_status[i], settings)

        points_status[i,0] = point_st
        if point_st != 1:
            output_v[i, 0, :] = v_tmp.flatten()
            output_A[:, :, i] = A_tmp
        else:
            output_v[i, 0, :] = v_g.flatten()
            output_A[:, :, i] = A_g            
    
    return output_v, output_A, points_status
    
    
def calc_d(ref_image, new_image, v, A, ref_points, H0, selection, F, win_idx, point_status, settings):
    # KLT distance calculation

    # Template of reference image (with ghost points for gradient calculation)
    template = ref_image[round(ref_points[1,0]-settings['win_size'][0]-1):round(ref_points[1,0]+settings['win_size'][0]+1), round(ref_points[0,0]-settings['win_size'][1]-1):round(ref_points[0,0]+settings['win_size'][1]+1)]

    # Find image gradient in ref_image
    #   TODO: to be correct derivative should be on new_image
    Ix, Iy = calc_gradient(template, settings['grad_method'], settings['grad_sigma'], settings['verbosityFlag']-1)

    # Remove ghost points
    Ix = Ix[1:-1, 1:-1]
    Iy = Iy[1:-1, 1:-1]
    template = template[1:-1, 1:-1]

    # Find image hessian in ref_image
    # -> This matrix should be invertible for a good solution (i.e., the image should contain enough gradient around the point to track)
    # -> A larger window is needed for full affine transformations compared to only translational 
    G = estimate_G(Ix, Iy, settings['win_size'][1], settings['win_size'][0], settings['motion_model'], settings['weighting'])
    G = np.transpose(H0)@G@H0

    if settings['condition_check']:
        condition_nr = np.linalg.cond(G)
        print(f'Condition number (G): {condition_nr}')

    # Newton-Raphson iterative calculation
    res_error = float('inf')
    for it in range(settings['iteration_criteria'][0]):
        # TODO: in first interation, only do translations

        # Warp the currect frame window
        J_win = warp_image(new_image, F, A, v, win_idx, settings['win_size'], settings['verbosityFlag']-1)

        # Image normalization
        if settings['image_normalization'] == 'none':
            # Do nothing
            pass
        elif settings['image_normalization'] == 'MeanVariance':
            delta_norm = np.mean(template) - np.mean(J_win)
            lambda_norm = np.var(template)/np.var(J_win)
            J_win = lambda_norm*J_win + delta_norm
    
        # Image mismatch vector
        e = estimate_e(template, J_win, Ix, Iy, settings['win_size'][1], settings['win_size'][0], settings['motion_model'], settings['weighting'])
        e = np.transpose(H0)@e

        # Residual motion vector
        b = np.linalg.solve(G, e)

        # Residual
        if settings['condition_check']:
            res_error_new = G@b-e
            if res_error_new > res_error:
                print('WARNING: affine LKOF is diverging!')
            res_error = res_error_new

        # Update of the tracking solution
        if settings['motion_model'] == 'translational':
                res = np.array([[A[0,0], A[0,1], v[0,0]], [A[1,0], A[1,1], v[1,0]], [0, 0, 1]])@np.array([[1, 0, b[0,0]], [0, 1, b[1,0]], [0, 0, 1]])
        elif settings['motion_model'] == 'affine':
            if settings['model_reduction'] == 'none':
                res = np.array([[A[0,0], A[0,1], v[0,0]], [A[1,0], A[1,1], v[1,0]], [0, 0, 1]])@np.array([[1+b[2,0], b[3,0], b[0,0]], [b[4,0], 1+b[5,0], b[1,0]], [0, 0, 1]])
            elif settings['model_reduction'] == 'full':
                res = np.array([[A[0,0], A[0,1], v[0,0]], [A[1,0], A[1,1], v[1,0]], [0, 0, 1]])@np.array([[1, 0, b[0,0]], [0, 1, b[1,0]], [0, 0, 1]])@np.array([[np.cos(b[2,0]), -sin(b[2,0]), 0], [sin(b[2,0]), cos(b[2,0]), 0], [0, 0, 1]])@np.array([[1, b[3,0], 0], [0, 1, 0], [0, 0, 1]])@np.array([[1+b[4,0], 0, 0], [0, 1+b[5,0], 0], [0, 0, 1]])
            else:
                d = np.zeros((6, 1))
                d[selection] = b
                res = np.array([[A[0,0], A[0,1], v[0,0]], [A[1,0], A[1,1], v[1,0]], [0, 0, 1]])@np.array([[1, 0, d[0,0]], [0, 1, d[1,0]], [0, 0, 1]])@np.array([[cos(d[2,0]), -sin(d[2,0]), 0], [sin(d[2,0]), cos(d[2,0]), 0], [0, 0, 1]])@np.array([[1, d[3,0], 0], [0, 1, 0], [0, 0, 1]])@np.array([[1+d[4,0], 0, 0], [0, 1+d[5,0], 0], [0, 0, 1]])

        v = res[0:2, [2]]
        A = res[0:2, 0:2]

        # Check if converged
        if np.linalg.norm(b[0:2,0]) <= settings['iteration_criteria'][1]:
            # print(f'Number of iterations: {it}')
            break
    
    # Detect occlusions
    if settings['occlusion_method'] != 'none':
        template_to_match = warp_image(new_image, F, res[0:2, 0:2], res[0:2, [2]], win_idx, settings['win_size'], settings['verbosityFlag']-1)
        occlusionFlag = detect_occlusion(template, template_to_match, settings['occlusion_method'], settings['occlusion_threshold'], settings['verbosityFlag'])
        if occlusionFlag:
            point_status = 1
    
    return v, A, point_status


def model_reduction_matrix(model_reduction, motion_method):
    # Model reduction matrix
    if model_reduction == 'none':
        if motion_method == 'translational':
            H0 = np.identity(2)
            selection = np.arange(2)
        elif motion_method == 'affine':
            H0 = np.identity(6)
            selection = np.arange(6)
    elif model_reduction == 'full':
        assert motion_method == 'affine', 'ERROR: model reduction is not allowed for a translational motion model!'
        H0 = np.array([[1, 0,  0, 0, 0, 0],
                       [0, 1,  0, 0, 0, 0],
                       [0, 0,  0, 0, 1, 0],
                       [0, 0, -1, 1, 0, 0],
                       [0, 0,  1, 0, 0, 0],
                       [0, 0,  0, 0, 0, 1]])
        selection = np.arange(6)
    else:
        assert motion_method == 'affine', 'ERROR: model reduction is not allowed for a translational motion model!'
        motion_components = model_reduction.split('+')
        
        selection = []
        for i in range(len(motion_components)):
            if motion_components[i] == 'translation':
                selection = sorted(list(set(selection) | set([0,1])))
            elif motion_components[i] == 'translationx':
                selection = sorted(list(set(selection) | set([0])))
            elif motion_components[i] == 'translationy':
                selection = sorted(list(set(selection) | set([1])))
            elif motion_components[i] == 'rotation':
                selection = sorted(list(set(selection) | set([2])))
            elif motion_components[i] == 'skew':
                selection = sorted(list(set(selection) | set([3])))
            elif motion_components[i] == 'scale':
                selection = sorted(list(set(selection) | set([4,5])))
            elif motion_components[i] == 'scalex':
                selection = sorted(list(set(selection) | set([4])))
            elif motion_components[i] == 'scaley':
                selection = sorted(list(set(selection) | set([5])))
        
        H0 = np.array([[1, 0,  0, 0, 0, 0],
                       [0, 1,  0, 0, 0, 0],
                       [0, 0,  0, 0, 1, 0],
                       [0, 0, -1, 1, 0, 0],
                       [0, 0,  1, 0, 0, 0],
                       [0, 0,  0, 0, 0, 1]])
        H0 = H0[:, selection]

    return H0, selection


def estimate_G(x_diff, y_diff, wx, wy, motion_model, W):
    # Estimating G, the Hessian matrix/spatial gradient matrix (6x6 matrix)
    # -> This matrix should not be (close to) singular to have good tracking
    x_diff = x_diff.flatten()
    y_diff = y_diff.flatten()
    
    if motion_model == 'translational':
        grads_T = np.array([x_diff, y_diff])
    elif motion_model == 'affine':
        x, y = np.meshgrid(np.arange(-wx,wx), np.arange(-wy, wy))

        grads_T = np.array([x_diff, y_diff, x.flatten()*x_diff, y.flatten()*x_diff, x.flatten()*y_diff, y.flatten()*y_diff])

    if W == 'none':
        G = grads_T@np.transpose(grads_T)
    else:
        G = grads_T@W@np.transpose(grads_T)

    return G


def estimate_e(I, J, Ix, Iy, wx, wy, motion_model, W):
    # Difference Function
    if motion_model == 'translational':
        D = I-J
        grads_T = np.array([Ix.flatten(), Iy.flatten()])
    if motion_model == 'affine':
        D = I-J
        x, y = np.meshgrid(np.arange(-wx,wx), np.arange(-wy,wy))
        grads_T = np.array([Ix.flatten(), Iy.flatten(), x.flatten()*Ix.flatten(), y.flatten()*Ix.flatten(), x.flatten()*Iy.flatten(), y.flatten()*Iy.flatten()])

    D = D.flatten()
    if W == 'none':
        e = grads_T@D
    else:
        wD = W*D
        e = grads_T@wD

    return np.transpose([e])