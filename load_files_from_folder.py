# Functions to load filesfrom a folder
#
# Contributors:
#   Thijs Willems
#
# KU Leuven, LMSD

import os
import re
from tkinter.filedialog import askdirectory

def load_files_from_folder(folder=None, sort=False, title='Select folder'):
    """
    Create a list with the file paths of all the files in a folder. If no folder specified, an 'Open dialog box' is shown to select the folder.

    Parameters
    ----------
    folder : str, default=None
        Folder path of the files to import. If "None", the 'Open dialog box' is shown.
    sort : boolean, default=False
        Determine if the filenames need to be sorted based on the number presend in the filename or not (e.g., for sorting frames of an image sequence).
    title : str, default='Select folder'
        Text to show on the directory selection window.

    Returns
    -------
    list
        A list with the file paths of the imported files.
    """
    filenames = []
    if folder is None:
        root = Tk()
        folder = askdirectory(parent=root, title=title)
        root.destroy()

    try:
        filenames = [os.path.join(folder, f).replace('\\', '/') for f in os.listdir(folder)]
    except FileNotFoundError:
        print('WARNING: files not found!')
        
    if sort:
        filenames = sort_files(filenames)
        
    return filenames

def sort_files(filenames):
    """
    Sort a list of filepaths according to the number in the filename (e.g., for sorting frames).

    Parameters
    ----------
    filenames : list of str
        Filenames to sort.

    Returns
    -------
    list
        A list with the sorted file paths of the imported files.
    """        
    # Sort the filenames
    frame_nrs = []
    for img_nr in range(len(filenames)):
        frame_nrs.append([int(s) for s in re.findall(r'\d+', filenames[img_nr])][-1])
    filenames_sorted = [filename for _, filename in sorted(zip(frame_nrs, filenames))]

    return filenames_sorted
