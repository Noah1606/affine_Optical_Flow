'''
Functions to load/save image data.

Contributors:
    Thijs Willems
    
KU Leuven, LMSD
'''

import pyMRAW
import cv2 as cv
import numpy as np
import imageio.v3 as iio
import os
from math import log, ceil
from tqdm import trange

from load_files_from_folder import load_files_from_folder
from data_conversion import change_color_scheme, change_bit_depth

def read_image(image, color_scheme='UNCHANGED', bit_depth='UNCHANGED'):
    """Read frame
    
    Read the frame form a filepaths/numpy.ndarray and returns the frame as an numpy.ndarray.
    
    Parameters
    ----------
    image : str or numpy.ndarray
        Filepath/numpy.ndarray to the image to read.
    color_scheme : str, default='UNCHANGED'
        Color scheme of the returned image (can be different than the original color scheme). Options: 'GRAY', 'BGR'
    bit_depth : str, default='UNCHANGED'
        Bit depth of the returned image (can be different than the original bit depth).
    
    Returns
    ------
    image : numpy.ndarray
        The image.
    """
    # Load the image as numpy.ndarray
    if isinstance(image, str):
        image = cv.imread(image, cv.IMREAD_UNCHANGED)

    # Check the color scheme
    if color_scheme != 'UNCHANGED':
        if len(image.shape) > 2 and color_scheme == 'GRAY':
            image = change_color_scheme(image, 'BGR', 'GRAY')
        elif len(image.shape) < 3 and color_scheme == 'BGR':
            image = change_color_scheme(image, 'GRAY', 'BGR')

    # Check bit denpth
    if bit_depth != 'UNCHANGED':
        image = change_bit_depth(image, bit_depth)    
    
    return image

def save_image(image, save_folder, name='img', output_type='PNG', compression='off', check_lossness=False):
    """
    Save the image into a specific format.

    Parameters
    ----------
    image : numpy.ndarray
        Image to save.
    output_type : str, default='PNG'
        Output file type of the images
            'BMP': image is converted to .bmp files (losless)
            'PNG': image is converted to .png files (lossless)
            'JPEG': image is converted to .jpg files
            'TIFF': image is converted to .tiff files (lossless)
            'WEBP': image is converted to .webp files (lossless, best compression)
    save_folder : str, default=''
        Path to the folder to save the image.
    name : str, default=''
        Name to save the image.
    compression : str, default='off'
        Determine if a data compression is done or not (may slow down the conversion, but can save storage space). Options: 'on' / 'off'.
    check_lossness : bool, default=False
        Check if the loaded converted image represents exactly the same data compared to the original
        image if True (this takes some time, so if you are sure or if this is not important, use the
        default value False)

    Returns
    -------
    output_path : str
        Path to the save image.
    """
    
    output_path = f'{save_folder}\\{name}'
   
    if output_type == 'BMP':
        # BMP: save raw image
        cv.imwrite(output_path + '.bmp', image)
        output  = output_path + '.bmp'
    elif output_type == 'PNG':
        # PNG: save image with lower compression—bigger file size but faster decoding (0, 9)
        if compression == 'on':
            cv.imwrite(output_path + '.png', image, [cv.IMWRITE_PNG_COMPRESSION, 9])  # Much slower
        else:
            cv.imwrite(output_path + '.png', image)  # Faster and also lossless
        output = output_path + '.png'
    elif output_type == 'JPEG':
        # JPEG: save image with lower quality—smaller file size (0, 100)
        cv.imwrite(output_path + '.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 100])
        output = output_path + '.jpg'
    elif output_type == 'TIFF':
        # TIFF:
        if compression == 'on':
            cv.imwrite(output_path + '.tiff', image, [cv.IMWRITE_TIFF_COMPRESSION])
        else:
            cv.imwrite(output_path + '.tiff', image)
        output = output_path + '.tiff'
    elif output_type == 'WEBP':
        # WebP:
        if compression == 'on':
            # Other method (slower, but little better compression)
            cwebp(image, output_path + '.webp', "-z 9")
            # cv.imwrite(output_path + '.webp', image, [cv.IMWRITE_WEBP_QUALITY, 101])
        else:
            cv.imwrite(output_path + '.webp', image)
        output = output_path + '.webp'
    else:
        raise ValueError('Please provide a valid output image type!')

    if check_lossness:
        # check that image saved and loaded again image is the same as original one
        if output_type == 'WEBP' and image.shape < 2:
            # .webp saves the images always with 3 channels
            saved_img = cv.imread(output, cv.IMREAD_GRAYSCALE)
        else:
            saved_img = cv.imread(output, cv.IMREAD_UNCHANGED)
        comparison = saved_img == image
        assert comparison.all()
    return output

def save_mraw(images, save_path, bit_depth=8, ext='mraw', info_dict={}):
    """
    Saves given sequence of images into .mraw file.
    Source: https://github.com/ladisk/pyMRAW

    Inputs:
    sequence : array_like of shape (n, h, w), sequence of `n` grayscale images
        of shape (h, w) to save.
    save_path : str, path to save the files. 
    bit_depth: int, bit depth of the image data. Currently supported bit depths are 8 and 16.
    ext : str, generated file extension ('mraw' or 'npy'). If set to 'mraw', it can be viewed in
        PFV. Defaults to '.mraw'.
    info_dict : dict, mraw video information to go into the .cih file. The info keys have to match
        .cih properties descriptions exactly (example common keys: 'Record Rate(fps)', 
        'Shutter Speed(s)', 'Comment Text' etc.).

    Outputs:
    mraw_path : str, path to output or .mraw (or .npy) file.
    cih_path : str, path to generated .cih file
    """

    filename, extension = path.splitext(save_path)
    mraw_path = '{:s}.{:s}'.format(filename, ext)
    cih_path = '{:s}.{:s}'.format(filename, '.cih')

    directory_path = path.split(save_path)[0]
    if not path.exists(directory_path):
        os.makedirs(directory_path)

    bit_depth_dtype_map = {
        8: np.uint8,
        16: np.uint16
    }
    if bit_depth not in bit_depth_dtype_map.keys():
        raise ValueError('Currently supported bit depths are 8 and 16.')
    
    if bit_depth < 16:
        effective_bit = bit_depth
    else:
        effective_bit = 12
    if np.max(images) > 2**bit_depth-1:
        raise ValueError(
            'The input image data does not match the selected bit depth. ' +
            'Consider normalizing the image data before saving.')

    # Generate .mraw file
    with open(mraw_path, 'wb') as file:
        for image in images:
            image = image.astype(bit_depth_dtype_map[bit_depth])
            image.tofile(file)
    file_shape = (int(len(images)), image.shape[0], image.shape[1])
    file_format = 'MRaw'

    image_info = {'Record Rate(fps)': '{:d}'.format(1),
                'Shutter Speed(s)': '{:.6f}'.format(1),
                'Total Frame': '{:d}'.format(file_shape[0]),
                'Original Total Frame': '{:d}'.format(file_shape[0]),
                'Start Frame': '{:d}'.format(0),
                'Image Width': '{:d}'.format(file_shape[2]),
                'Image Height': '{:d}'.format(file_shape[1]),
                'Color Type': 'Mono', 
                'Color Bit': bit_depth,
                'File Format' : file_format,
                'EffectiveBit Depth': effective_bit,
                'Comment Text': 'Generated sequence. Modify measurement info in created .cih file if necessary.',
                'EffectiveBit Side': 'Lower'}

    image_info.update(info_dict)

    cih_path = '{:s}.{:s}'.format(filename, 'cih')
    with open(cih_path, 'w') as file:
        file.write('#Camera Information Header\n')
        for key in image_info.keys():
            file.write('{:s} : {:s}\n'.format(key, str(image_info[key])))
    
    return mraw_path, cih_path

def export_MRAW(file_cih, save_folder, frame_nr=None):
    # Read Photron MRAW image sequence and save the individual frames into a folder.
    # Based on the pyMRAW package (https://github.com/ladisk/pyMRAW).
    
    cih = pyMRAW.get_cih(file_cih)
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']

    file_mraw, ext = path.splitext(file_cih)
    mraw = open(file_mraw + '.mraw', 'rb')
    mraw.seek(0, 0)  # find the beginning of the file
    image_data = pyMRAW.load_images(mraw, h, w, N)  # load N images
    mraw.close()
    
    if frame_nr is None:
        for i in trange(N, desc='Saving all frames'):
            image = image_data[:, :, i]
            
            # cv.imshow('Loaded image', image)
            # cv.waitKey(0)

            name = f"img{str(i).zfill(ceil(log(N, 10)))}"
            save_image(image, save_folder, name)
    else:
        image = image_data[:, :, frame_nr]
        
        name = f"img{str(frame_nr).zfill(ceil(log(N, 10)))}"
        save_image(image, save_folder, name)

class imageStream:
    # imageStream class
    def __init__(self, name='ImageStream', data=None, fps=None, color_scheme='UNCHANGED', bit_depth='UNCHANGED'):
        self.name = name
        self.type = None
        self.stream_location = None
        self.images = None
        self.nr_images = 0
        self.fps = fps
        self.time = None
        self.image_size = (None, None)  # (w, h)
        self.idx = 0
        self.color_scheme = color_scheme
        self.bit_depth = bit_depth
        
        if data is not None:
            self.set_data(data)
    
    def set_data(self, data):
        if os.path.isfile(data):
            # file exists
            file, ext = os.path.splitext(data)
            if ext in ['.cih', '.cihx']:
                # Based on the pyMRAW package (https://github.com/ladisk/pyMRAW).
                self.type = 'photron'
                self.stream_location = [data, file + '.mraw']
                
                metadata = pyMRAW.get_cih(self.stream_location[0])
                self.nr_images = metadata['Total Frame']
                self.image_size = (metadata['Image Width'], metadata['Image Height'])
                self.fps = metadata['Record Rate(fps)']
                self.time = np.arange(0.0, self.nr_images*1/self.fps, 1/self.fps, dtype=np.float64)

                mraw = open(self.stream_location[1], 'rb')
                mraw.seek(0, 0)  # find the beginning of the file
                self.images = pyMRAW.load_images(mraw, self.image_size[1], self.image_size[0], self.nr_images)
                mraw.close()
            if ext in ['.cine']:
                # See also: https://imageio.readthedocs.io/en/stable/index.html, https://github.com/ottomatic-io/pycine
                #TODO: read as memmap
                self.type = 'phantom'
                self.stream_location = [data, file + '.cine']
                
                metadata1 = iio.immeta(self.stream_location[0])
                metadata2 = iio.improps(self.stream_location[0])
                self.nr_images = int(metadata1['duration']*metadata1['fps'])
                self.image_size = (metadata2.shape[2], metadata2.shape[1])
                self.fps = metadata1['fps']
                self.time = np.arange(0.0, self.nr_images*1/self.fps, 1/self.fps, dtype=np.float64)

                self.images = iio.imiter(self.stream_location[0], plugin="pyav")
        elif os.path.isdir(data):
            # directory exists
            self.type = 'folder'
            self.stream_location = data
            self.images = load_files_from_folder(self.stream_location, sort=True)
            self.nr_images = len(self.images)
            if self.fps is not None:
                self.time = np.arange(0.0, self.nr_images*1/self.fps, 1/self.fps, dtype=np.float64)
        else:
            raise FileNotFoundError('Image data not found!')

    def __len__(self):
        return self.nr_images

    def __getitem__(self, i):
        if self.type == 'folder':
            image = read_image(self.images[i], color_scheme=self.color_scheme, bit_depth=self.bit_depth)
        elif self.type == 'photron':
            image = read_image(self.images[:, :, i], color_scheme=self.color_scheme, bit_depth=self.bit_depth)
        elif self.type == 'phantom':
            image_temp = iio.imread(self.stream_location[0], index=i, plugin="pyav")
            image = read_image(image_temp, color_scheme=self.color_scheme, bit_depth=self.bit_depth)
        return image

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.nr_images:
            image = self.__getitem__(self.idx)
            self.idx += 1
            return self.idx-1, image
        else:
            self.idx = 0
            raise StopIteration
        
    def image_array(self, image_range=(0, -1)):
        """Output an array with all the images in the given range. If the range is not specified, return all the images.
        
        WARNING: calling this function can result in an out-of-memory!
        """
        images = np.zeros((self.nr_images, self.image_size[1], self.image_size[0]))
        
        start_i = image_range[0]
        if image_range[1] == -1:
            stop_i = self.nr_images
        else:
            stop_i = image_range[1]
            
        for i in range(start_i, stop_i, 1):
            images[i, :, :] = self[i]
        return images
    
    def save(self, image_number, save_folder, name):
        save_image(self[image_number], save_folder, name, output_type='BMP', compression='off', check_lossness=False)
