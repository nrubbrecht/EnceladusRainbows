# This file contains the definitions used in the notebooks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
import sys
from pyvims import VIMS
import spiceypy as spice
import os
from datetime import datetime, timedelta
from scipy.ndimage import rotate
from skimage.transform import rescale

# -------------------- VIMS processing --------------------------
# Function to calculate percentiles for color limits
def calculate_percentiles(data, lower=2, upper=98):
    lower_limit = np.percentile(data, lower)
    upper_limit = np.percentile(data, upper)
    return lower_limit, upper_limit

def adjust_aspect_ratio(image, aspect_ratio):
    scaled_image = rescale(image, (aspect_ratio, 1), anti_aliasing=True)
    return scaled_image

def align_stripes_with_mask(image, angle):
    rotated_image = rotate(image, angle, reshape=False)
    mask = rotated_image != 0  # Mask identifying non-zero elements
    return rotated_image, mask