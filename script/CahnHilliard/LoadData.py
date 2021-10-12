import sys, os, glob
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from utilities.Exports import exportVTK

################################

class LoadData: #input_folder, ndim=3, export_npy=True, export_vtk=False):

    def __init__(self, path, ext=None):
        mask = path + "/*"
        if ext: mask += "." + ext
        self.filenames = sorted(glob.glob(mask), key=key_func)
        self.size = len(self.filenames)

    def __getitem__(self, i):
        return np.load(self.filenames[i])


################################
#       Helper functions
################################

def key_func(filename_with_path):
    filename = os.path.split(filename_with_path)[1]
    filename_without_ext = os.path.splitext(filename)[0]
    num = re.findall('\d+', filename_without_ext)[0]
    return int(num)
