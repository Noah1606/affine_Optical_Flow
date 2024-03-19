"""
Routines to perform feature tracking.

Contributors: 
    Thijs Willems
    
KU Leuven, LMSD
"""
import statistics
import time
import keyboard

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from image_data import imageStream
from affine_optical_flow import calc_affine_of

#TODO: frame averaging for better noise robustness
class Tracker:
    """
    Camera tracking class implementing different tracking methods:
        - Lucas-Kanade Optical flow
            - OpenCV implementation (both CPU or GPU): rather efficient + possibility to use pyramids
            - Direct least-squares solution of the equations: less efficient + no possibility to use pyramids, but fully adjustable
        - Affine optical flow
        - Simplified optical flow

    Attributes
    ----------
    images
    feature_points
    nr_frames
    motion_method
    tracking_method
    tracking_param
    tracks
        TODO: make this a 'Motion' data class
        The tracked positions of the feature points in all the frames as a list of length=#frames and containing
        a list of pixel coordinates (numpy.ndarrays of shape=(#points, 1, 2(u,v))) (one for each ROI).
    marker_grouping
        List of #frames elements which are lists of 1D numpy.ndarrays (one for each ROI) of lenght=#feature
        points indicating to which feature the feature point is belonging.
    valid
        List of #rois elements which are a list of #feature_points elements indicating which feature points are correctly tracked 
        and thus retained in the 'tracks' attribute.

    Methods
    -------
    set_images(images)
        Define the images. This overrides already added images.
    set_feature_points(feature_points)
        Define the initial feature points to track. This overrides already defined feature points.
    set_tracking_method(tracking_method, param)
        Define the tracking method to use with its parameters.
    calc()
        Perform the tracking with the specified tracking method.
    plot()
        Plot the tracking results:
            -> a plot with the feature points showing the tracked (in green) and discarded (in red) feature points;
            -> a animation with the position of the tracked feature point in the successive frames.
    """
    def __init__(self, images=None, feature_points=None, nr_frames=-1, motion_method='absolute', tracking_method='aof', **kwargs):
        """
        Initialization of the Tracking class.

        Parameters
        ----------
        images : imagestream, default=None
            Images to track the feature points in.
        feature_points : list of list of numpy.ndarray, default=None
            User defined (subpixel) points to track as a list of length=1 and containing
            a list of pixel coordinates (numpy.ndarrays of shape=(#points, 1, 2(u,v)))
            (one for each ROI).
        nr_frames : int or -1, defailt=-1
            Number of frames to track the motion in. 
                If < images.nr_images: only the specified number of frames, starting from the first frame, are used for tracking.
                If > images.nr_images: all the frames in the ImageSequence are used for tracking (equal to setting nr_frames=-1).
                If -1: all the frames in the ImageSequence are used for tracking.
        motion_method : str, default='absolute'
            Motion reference method to use:
                'absolute': uses the first frame as a reference frame for each frame (static reference), this can help reducing
                            error accumulation in the tracking but requires the method to track larger displacements;
                'incremental': uses the previous frame as a reference frame for the next frame (changing reference), this
                               can lead to error accumulation but the displacement to track between the successive frames
                               is smaller.
        tracking_method : str, default='aof'
            See: set_tracking_method

        Keyword arguments
        -----------------
        Tracking method parameters as keyword arguments -> See: set_tracking_method

        """
        self.images = None
        self.feature_points = None
        self.nr_frames = nr_frames
        self.motion_method = motion_method
        self.tracking_method = None
        self.tracking_param = None
        self.tracks = None
        self.grouping = None
        self.valid = None

        if images is not None:
            self.set_images(images)
        if feature_points is not None:
            self.set_feature_points(feature_points)
        if tracking_method is not None:
            self.set_tracking_method(tracking_method, kwargs)

    def set_images(self, images):
        """
        Define the images. This overrides already added images.

        Parameters
        ----------
        images : ImageSequence class instance.

        Returns
        -------
        /
        """
        if isinstance(images, imageStream):
            self.images = images
        else:
            self.images = imageStream(name='ImageStream', data=images, fps=None, color_scheme='GRAY', bit_depth='UNCHANGED')        

    def set_feature_points(self, feature_points):
        """
        Define the initial feature points to track. This overrides already defined feature points.

        Parameters
        ----------
        feature_points : list of list of numpy.ndarray, default=None
            User defined (subpixel) points to track as an array of pixel coordinates (numpy.ndarrays of shape=(#points, 1, 2(u,v))).

        Returns
        -------
        /
        """
        self.feature_points = feature_points

    def set_tracking_method(self, tracking_method, param):
        """
        Define the tracking method to use with its parameters.

        Parameters
        ----------
        tracking_method : str, default='aof'
            Motion tracking method to use:
                'aof': own affine optical flow implementation.
        param : dict
            Tracking method parameters as keyword arguments

        Returns
        -------
        /
        """
        self.tracking_method = tracking_method

        if self.tracking_method == 'aof':
            if 'aof_params' not in param:
                param['aof_params'] = None
            if 'initial_guess_function' not in param:
                param['initial_guess_function'] = None
            if 'backtracking' not in param:
                param['backtracking'] = None
            if 'reject_param' not in param:
                param['reject_param'] = 0.0
            if 'grouping' not in param:
                param['grouping'] = None
            if 'fps' not in param:
                param['fps'] = 60
        self.tracking_param = param

    def calc(self):
        """
        Perform the tracking with the specified tracking method.
        """
        tracks = np.zeros((self.feature_points.shape[0], self.images.nr_images, 2))
        for img_nr, img in tqdm(self.images, f'Tracking (Method: {self.tracking_method})'):
            if (self.nr_frames > 0) and (img_nr > self.nr_frames):
                break
            if img_nr == 0:
                # take first frame
                prev_gray = img
                start_p = self.feature_points[:, [img_nr], :]
                tracks[:, [img_nr], :] = start_p
                valid = np.ones((self.feature_points.shape[0], self.images.nr_images)) * True  # input points are valid by default!

                if self.motion_method == 'absolute':
                    start_gray = prev_gray
                continue

            if self.motion_method == 'incremental':
                img0, img1 = prev_gray, img
                p0 = tracks[:, [img_nr-1], :]
            elif self.motion_method == 'absolute':
                img0, img1 = start_gray, img
                p0 = start_p

            if len(p0) == 0:
                break

            if self.tracking_param['initial_guess_function'] is not None:
                p_guess = self.tracking_param['initial_guess_function'](img_nr)
            else:
                p_guess = None
            
            if self.tracking_method == 'aof':
                p1, A1, st1 = calc_affine_of(img0, img1, p0, initial_guess=p_guess, settings=self.tracking_param['aof_params'])
          
            tracks[:, [img_nr], :] = p1
            valid[:, [img_nr]] = st1
            prev_gray = img

        self.tracks = tracks
        self.valid = valid        

    def plot(self, savepath=None, lineSwitch=True):
        """
        Plot the tracking results:
            -> a plot with the feature points showing the tracked (in green) and discarded (in red) feature points;
            -> a plot with the u- and v-displacements of the feature points in time;
            -> a animation with the position of the tracked feature point in the successive frames.
            
        Parameters
        ----------
        savepath : str or None, default=None
            Path to the folder to save the plots. If None, nothing is saved.
        lineSwitch : boolean, default=True
            Determine if track lines are shown in the animation or not.
        """
        # Animation
        #TODO: add controls and save option (not working properly yet)
        # vis_list = []
        tracks_mat = signproc.roi_to_img(self.tracks)
        marker_thickness = 2
        marker_size = 5
        colors = np.random.randint(0, 255, (tracks_mat[0].shape[0], 3))
        
        # plt.style.use("ggplot")
        plt.ion()
        fig, ax = plt.subplots()
        fig.canvas.draw()
        
        if savepath is not None:
            out = cv.VideoWriter(savepath + '/Tracks.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, self.images.size)
            # out = cv.VideoWriter(savepath + '/Tracks.mp4', cv.VideoWriter_fourcc('M','J','P','G'), self.tracking_param['fps'], (1080, 1920))
        pause = False
        status = ''
        
        tracks_prev = tracks_mat[0].reshape(-1, 2).astype(int)
        img_nr = 0
        while img_nr <= self.nr_frames:
            tracks_new = tracks_mat[img_nr].reshape(-1, 2).astype(int)
            vis = cv.cvtColor(self.images[img_nr, 'array', 'unchanged'].copy(), cv.COLOR_GRAY2RGB)
            if img_nr == 0:
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                mask = np.zeros_like(vis)
            for i, (new, prev) in enumerate(zip(tracks_new, tracks_prev)):
                a, b = new.ravel()
                c, d = prev.ravel()
                if lineSwitch:
                    cv.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                    cv.drawMarker(vis, (a, b), colors[i].tolist(), cv.MARKER_CROSS, marker_size, marker_thickness)
                else:
                    cv.drawMarker(vis, (a, b), (255, 0, 0), cv.MARKER_CROSS, marker_size, marker_thickness)                    
            # vis_list.append(vis)
            viss = cv.add(vis, mask)
            if savepath is not None:
                out.write(viss)

            ax.clear()  # erase previous plot
            ax.imshow(viss)  # create plot
            ax.set_title(f'Tracking (frame: {img_nr}){status}')
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            time.sleep(0.1)
            if keyboard.is_pressed('p'):
                pause = True
            tracks_prev = tracks_new

            if not pause:
                img_nr += 1
                status = ''
            else:
                status = ' - PAUSED'
                time.sleep(0.1)
                if keyboard.is_pressed('right'):
                    img_nr += 1
                elif keyboard.is_pressed('left'):
                    img_nr -= 1
                elif keyboard.is_pressed('space'):
                    pause = False

        # make sure that the last plot is kept
        plt.ioff()
        plt.show()
        if savepath is not None:
            out.release()

        # Accepted and discarted feature points
        vis = cv.cvtColor(self.images[0, 'array', 'unchanged'].copy(), cv.COLOR_GRAY2RGB)
        for roi_nr in range(self.images.nr_rois):
            u_roi, v_roi, _, _ = self.images.rois[roi_nr]
            for u, v in np.float32(self.feature_points[0][roi_nr]).reshape(-1, 2):
                vis[int(v+v_roi), int(u+u_roi), :] = [255, 0, 0]
            for u, v in np.float32(self.tracks[0][roi_nr]).reshape(-1, 2):
                vis[int(v+v_roi), int(u+u_roi), :] = [0, 255, 0]
        fig = plt.figure()
        plt.title('Feature points (RED: discarded, GREEN: tracked)')
        plt.imshow(vis)
        if savepath is not None:
            plt.savefig(savepath + '/TrackedPointsStatus.png', bbox_inches='tight')
        plt.show()
