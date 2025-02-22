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
import imageio

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

def pixels_in_rectangle(array, lower_left, upper_right):
    """
    Finds the pixels in a 2D array that fall within the specified rectangle.

    Parameters:
    - array: 2D array representing the image.
    - lower_left: Tuple (x, y) representing the lower left corner of the rectangle.
    - upper_right: Tuple (x, y) representing the upper right corner of the rectangle.

    Returns:
    - NumPy array containing the coordinates (x, y) of pixels within the rectangle.
    """
    pixels = []
    ll_col, ll_row = lower_left
    ur_col, ur_row = upper_right

    for row in range(ur_row, ll_row + 1):
        for col in range(ll_col, ur_col + 1):
            if 0 <= row < len(array)+1 and 0 <= col < len(array[0]+1):
                pixels.append([col, row])

    return np.array(pixels)


def calculate_average_spectrum(positions, cube_input):
    """
    Calculate the average spectrum from a list of positions.

    Parameters:
    - positions: List of positions.
    - cube: The cube object used to calculate the spectrum.

    Returns:
    - Average spectrum.
    """
    summed_spectrum = None

    for i in range(len(positions)):
        position_i = cube_input@(int(positions[i, 0]), int(positions[i, 1]))
        spectrum_i = np.copy(position_i.spectrum)  # Copy the spectrum data to avoid in-place modification

        if summed_spectrum is None:
            summed_spectrum = spectrum_i
        else:
            summed_spectrum += spectrum_i

    return summed_spectrum / len(positions)


def moving_average(data, window_size):
    """
    Apply moving average smoothing to a series of data.

    Parameters:
        data (array-like): The input data series.
        window_size (int): The size of the moving average window.

    Returns:
        smoothed_data (ndarray): The smoothed data series.
    """
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data


def create_video_from_directory(input_dir, output_file, num_frames,  fps=10):
    # Get list of image files in the directory
    # image_files = [f"frame_{i}.png" for i in range(1, num_frames+1)]
    image_files = [os.path.join(input_dir, f"frame_{i}.png") for i in range(0, num_frames)]

    # Read images and append them to a list
    frames = []
    for file in image_files:
        frames.append(imageio.v2.imread(file))

    # Create video from frames
    imageio.mimsave(output_file, frames, fps=fps)


def bresenham_line(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    line = []
    x, y = x1, y1
    while True:
        line.append([x, y])
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(line)


def find_true_indices(matrix):
    indices = np.where(matrix)
    true_indices = [[int(row + 1), int(col + 1)] for row, col in zip(indices[1], indices[0])]
    return np.array(true_indices)


def remove_bad_pix(cube):
    # remove hot pixels and blocking ranges
    # hot pixels: 1.24 μm, 1.33 μm, 3.23 μm, 3.24 μm, 3.83 μm
    # IR focal plane blocking filters: 1.6–1.67 μm, 2.94–3.01 μm and 3.83–3.89 μm
    hot = cube.w_hot_pixels()
    indices2del = []
    for hotpix in hot:
        index = cube._wvln(hotpix)
        indices2del.append(index)
    indices2del = np.array(indices2del)
    # get indices for filter blocking ranges
    block_indices = [[cube._wvln(1.6), cube._wvln(1.67)],
                     [cube._wvln(2.94), cube._wvln(3.01)],
                     [cube._wvln(3.83), cube._wvln(3.93)]]
    arrays = [np.arange(start, end + 1) for start, end in block_indices]
    # Concatenate the arrays into a single 1D NumPy array
    blocks = np.concatenate(arrays)
    indices2del = np.unique(np.sort(np.hstack((indices2del, blocks))))  # indices that need to be removed

    # Create a mask where True indicates elements to keep
    mask = np.ones(len(cube.wvlns), dtype=bool)
    mask[indices2del] = False
    return mask


def select_true_box(shape, top_left, bottom_right):
    """

    Parameters
    ----------
    shape (rows, columns)
    top_left (row, col)
    bottom_right (row, col)

    Returns
    -------
    np.array mask with true values in box
    """
    # Create an array filled with False values
    arr = np.zeros(shape, dtype=bool)

    # Extract coordinates of the rectangle
    top, left = top_left
    bottom, right = bottom_right

    # Assign True values to the rectangle
    arr[top:bottom, left:right] = True

    return arr


