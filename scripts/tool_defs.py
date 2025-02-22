import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import scipy.special as scp
from scipy import signal
import pandas as pd
import plotly.graph_objects as go
import spiceypy as spice
import csv
from datetime import datetime, timedelta
import copy
import plotly.io as pio
import os
from matplotlib.pyplot import figure
import math
from plotly.colors import sample_colorscale
from pyvims import VIMS
from matplotlib.path import Path
from matplotlib.colors import Normalize


def get_time_flyby(orbit_or_encounter, file_path, time_range=None):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        closest_approach = None
        closest_date_time = None
        for row in reader:
            orbit, encounter, date_time_str = row
            if orbit == orbit_or_encounter or encounter == orbit_or_encounter:
                date_time = datetime.strptime(date_time_str, '%Y/%m/%d %H:%M')
                if closest_approach is None or (date_time - closest_approach) < timedelta(0):
                    closest_approach = date_time
                    closest_date_time = date_time

    if closest_date_time is not None:
        if time_range:
            start_time = closest_date_time - timedelta(hours=time_range)
            end_time = closest_date_time + timedelta(hours=time_range)
            return [start_time.strftime('%Y/%m/%d %H:%M'), end_time.strftime('%Y/%m/%d %H:%M')]
        else:
            return [closest_date_time.strftime('%Y/%m/%d %H:%M')]


def get_remaining_ets(flyby_name, step_size, start_et=None, end_et=None):
    """
    Extracts Ephemeris Time (ET) values from a text file containing date intervals
    and returns the ET values that fall within any of the intervals.

    Parameters:
    - filename: String representing the path to the text file.
    - step_size: Integer representing the step size for generating the range of ET values.

    Returns:
    - remaining_ets: List of ET values that fall within any of the intervals.
    """

    filename = '../data/kernels/ck/{}.txt'.format(flyby_name)
    # Read the text file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract dates and convert to ET
    ets = []
    for line in lines:
        start_date, end_date = line.strip().split()
        et_start = spice.str2et(start_date)
        et_end = spice.str2et(end_date)
        ets.append((et_start, et_end))

    # Generate a range of ET values with the specified step size
    if start_et:
        min_et = start_et
    else:
        min_et = min(et[0] for et in ets)
    if end_et:
        max_et =end_et
    else:
        max_et = max(et[1] for et in ets)

    et_range = range(int(min_et), int(max_et), step_size)

    # Filter out ET values not within any interval
    filtered_ets = []
    for et in et_range:
        for interval in ets:
            if interval[0] <= et <= interval[1]:
                filtered_ets.append(et)
                break

    # Return the remaining ET values
    return sorted(filtered_ets)  # Optional: Sort the ET values if needed


def shift(x, y, z, x_shift=None, y_shift=None, z_shift=None):
    if x_shift:
        x = x + x_shift
    if y_shift:
        y = y + y_shift
    if z_shift:
        z = z + z_shift
    return x, y, z


def angle_between_vectors(vec1, vec2):
    # Normalize vectors
    vec1_normalized = vec1 / np.linalg.norm(vec1)
    vec2_normalized = vec2 / np.linalg.norm(vec2)

    # Compute dot product
    dot_product = np.dot(vec1_normalized, vec2_normalized)

    # Compute angle (in radians)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def sphere(sphere_radius, x_shift=None, y_shift=None, z_shift=None):
    # Create sphere data
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    x = x * sphere_radius
    y = y * sphere_radius
    z = z * sphere_radius
    if x_shift:
        x = x + x_shift
    if y_shift:
        y = y + y_shift
    if z_shift:
        z = z + z_shift
    return x , y, z


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rotate_x(x, y, z, angle):
    angle = angle/180 * np.pi
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle), -np.sin(angle)],
                                [0, np.sin(angle), np.cos(angle)]])
    rotated_vertices = np.dot(rotation_matrix, np.array([x.flatten(), y.flatten(), z.flatten()]))
    return rotated_vertices[0].reshape(x.shape), rotated_vertices[1].reshape(y.shape), rotated_vertices[2].reshape(z.shape)


def rotate_y(x, y, z, angle):
    angle = angle/180 * np.pi
    rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                [0, 1, 0],
                                [-np.sin(angle), 0, np.cos(angle)]])
    rotated_vertices = np.dot(rotation_matrix, np.array([x.flatten(), y.flatten(), z.flatten()]))
    return rotated_vertices[0].reshape(x.shape), rotated_vertices[1].reshape(y.shape), rotated_vertices[2].reshape(z.shape)


def rotate_z(x, y, z, angle):
    angle = angle/180 * np.pi
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    rotated_vertices = np.dot(rotation_matrix, np.array([x.flatten(), y.flatten(), z.flatten()]))
    return rotated_vertices[0].reshape(x.shape), rotated_vertices[1].reshape(y.shape), rotated_vertices[2].reshape(z.shape)


def cone(cone_height, scattering_angle, direction_vector=[0,0,0], grid_res=None, x_rotate=None, y_rotate=None, z_rotate=None, x_shift=None, y_shift=None,
         z_shift=None):

    cone_rad = np.tan(scattering_angle * np.pi / 180)

    # cone_rad = 1
    if grid_res is None:
        grid_res = [40, 40]
    # create mesh
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:40j]
    # unit cone
    x = np.cos(u) * np.sin(v) * cone_height
    y = np.sin(u) * np.sin(v) * cone_height
    z = f(x, y)
    # add dimensions
    x = cone_rad * x
    y = cone_rad * y
    z = -z

    # algin with [1, 0,0]
    x, y, z = rotate_y(x, y, z, -90)
    if direction_vector==[0,0,0]:
        x, y, z = x, y, z
    else:
        # transform vector to declination and right ascension on celestial sphere
        projection_xy = np.array([direction_vector[0], direction_vector[1], 0])
        declination = angle_between_vectors(projection_xy, np.array(direction_vector))
        # adjust sign
        if direction_vector[2] <= 0:
            declination = -declination

        right_ascension = angle_between_vectors(np.array([1, 0, 0]), projection_xy)
        if direction_vector[1] <= 0:
            right_ascension = -right_ascension

        x, y, z = rotate_y(x, y, z, -declination)
        x, y, z = rotate_z(x, y, z, right_ascension)

    # rotations
    if x_rotate:
        x, y, z = rotate_x(x, y, z, x_rotate)
    if y_rotate:
        x, y, z = rotate_y(x, y, z, y_rotate)
    if z_rotate:
        x, y, z = rotate_z(x, y, z, z_rotate)

    # shift tip
    if x_shift:
        x = x + x_shift
    if y_shift:
        y = y + y_shift
    if z_shift:
        z = z + z_shift
    return x, y, z


def plot_pyramid_surface(vertices):
    """
    Plot a surface representation of a pyramid defined by its five points.

    Parameters:
    - vertices: List of 5 3-dimensional vectors representing the points of the pyramid.
                The first four points should form the base, the last one the tip.

    Returns:
    - List of Plotly trace objects representing the vertices and faces of the pyramid.
    """
    v = vertices
    # Generate list of sides' polygons of the pyramid
    faces = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
             [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

    # Extract x, y, z coordinates of each vertex
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # Define traces for vertices and faces
    trace_vertices = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='blue'))
    traces_faces = []
    for face in faces:
        x_face = [vertex[0] for vertex in face]
        y_face = [vertex[1] for vertex in face]
        z_face = [vertex[2] for vertex in face]
        trace_face = go.Mesh3d(x=x_face, y=y_face, z=z_face, color='cyan', opacity=0.25)
        traces_faces.append(trace_face)

    return traces_faces


def select_lambdas(array, targets):
    df = pd.DataFrame()
    for target in targets:
        # Calculate the absolute differences between each row's first entry and the target
        differences = (array["lambda"] - target).abs()
        # Find the index of the row with the minimum absolute difference
        closest_row_index = differences.idxmin()
        # Return the row with the minimum absolute difference
        row = array.iloc[[closest_row_index]]
        # print(row)
        # df = df.append(row)
        df = pd.concat([df, row])

    return df


def geodetic2geocentric_lat(geod_lat, a, b):
    geod_lat = geod_lat*np.pi/180
    flattening = 1-b/a
    geocen_lat = np.arctan((1-flattening)**2*np.tan(geod_lat))
    return geocen_lat*180/np.pi


def f(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def get_circle_angle(x, y):
    alpha = np.zeros_like(x)  # Initialize array to store angles

    # Calculate angles for each point in terms of quadrants
    mask1 = (x > 0) & (y > 0)
    mask2 = (x < 0) & (y > 0)
    mask3 = (x < 0) & (y < 0)
    mask4 = (x > 0) & (y < 0)

    alpha[mask1] = np.arctan(y[mask1] / x[mask1])
    alpha[mask2] = np.arctan(y[mask2] / -x[mask2]) + np.pi / 2
    alpha[mask3] = np.arctan(y[mask3] / x[mask3]) + np.pi
    alpha[mask4] = np.arctan(-y[mask4] / x[mask4]) + 3 * np.pi / 2

    return alpha


def generate_circle_points(center, radius, direction, num_points=100, origin=np.array([0,0,0])):
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Vector from origin to the center of the circle
    vec_to_center = center - origin
    vec_to_center = vec_to_center / np.linalg.norm(vec_to_center)

    # Ensure the orthogonal vectors are indeed orthogonal
    assert np.abs(np.dot(direction, vec_to_center)) < 1e-2, "Direction and vec_to_center are not orthogonal"

    # Generate points on the circle using parametric equations
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([center + radius * (np.cos(t) * vec_to_center + np.sin(t) * direction) for t in theta])

    return circle_points


def filter_circle_points(circle_points, cs):
    # Calculate distances from the origin (0, 0, 0)
    distances = np.linalg.norm(circle_points, axis=1)
    # Filter points based on the distance
    filtered_points = circle_points[distances <= cs]
    return filtered_points


# Function to convert a grayscale color to an RGB string
def grayscale_to_rgb_string(grayscale):
    value = int(grayscale[0] * 255)
    return f'rgb({value},{value},{value})'


# Function to create a mesh for each pixel
def create_pixel_mesh(pixel_corner_array, color):
    return go.Mesh3d(
        x=pixel_corner_array[:,0],
        y=pixel_corner_array[:,1],
        z=pixel_corner_array[:,2],  # You can replace this with actual Z values if needed
        color=grayscale_to_rgb_string(color),
        opacity=1,
        flatshading=True,
        showscale=False,
        i=[0, 1, 2, 3],
        j=[1, 2, 3, 0],
        k=[2, 3, 0, 1]
    )


def stereographic_projection(theta, phi, R=1):
    # Stereographic projection function
    # theta is elevation, phi right ascension
    x = R * np.cos(theta) * np.sin(phi) / (1 + np.sin(theta))
    y = R * np.sin(theta) / (1 + np.sin(theta))
    return x, y


def extract_value_from_label(label_file_path, variable_name):
    value = None
    with open(label_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(variable_name):
                # Split the line on '=' and strip to get the value
                value = line.split('=')[1].strip()
                break
    return value


def get_wavelengths_and_colors(input_str):

    if input_str == "vis":
        central_wavelengths = [400, 475, 530, 575, 600, 700]
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    elif input_str.isdigit():
        central_wavelengths = [int(input_str)]
        colors = ['blue']
    elif input_str == "rgb":
        central_wavelengths = [440, 510, 650]
        colors = ['blue', 'green', 'red']
    elif input_str == "red":
        central_wavelengths = [700]
        colors = ['red']
    elif input_str == "purple":
        central_wavelengths = [400]
        colors = ['purple']
    elif input_str == "blue":
        central_wavelengths = [475]
        colors = ['blue']
    elif input_str == "green":
        central_wavelengths = [530]
        colors = ['green']
    elif input_str == "nir":
        central_wavelengths = [900, 1000, 1100, 1200]
        colors = ['yellow', 'gold', 'orange', 'red']
    elif input_str == "vis_nir":
        central_wavelengths = [475, 530, 700, 900, 1100, 1300]
        colors = ['blue', 'green', 'red', 'darkred', 'lightgrey', 'dimgrey']
    elif input_str == "vis_swir":
        central_wavelengths = [400, 700, 1100, 1600, 2100, 2600]
        colors = ['purple', 'red', "darkred", "lightgrey", "darkgrey","dimgrey"]
    elif input_str == "vis_swir_cont":
        central_wavelengths = [400, 950, 1500, 2050, 2600]
        colors = ['purple', 'red', "darkred", "lightgrey","dimgrey"]
    elif input_str == "swir":
        central_wavelengths = [1100, 1400, 1700, 2000, 2300, 2600]
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    elif input_str == "full_vims":
        central_wavelengths = [400, 650, 1000, 2000, 4000, 5000]
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    elif input_str == "vims_vis":
        central_wavelengths = [350, 450, 550, 700, 850, 1000]
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    elif input_str == "vims_ir":
        central_wavelengths = [850, 1000, 2000, 2500, 4000, 5000]
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    elif input_str == "custom":
        central_wavelengths = [2700, 2900, 3100, 3300, 3500]
        colors = ["blue", 'green', 'yellow', 'orange', 'red']
    elif input_str == "vims_check":
        central_wavelengths = [350, 1000]
        colors = ['purple', 'red']
    elif input_str == "iss":
        central_wavelengths = [200, 600, 1000]
        colors = ['purple','green' , 'red']
    else:
        # If input_str is not predefined, return empty lists
        central_wavelengths = []
        colors = []

    print("Selected wavelengths [nm]:", central_wavelengths)
    print("Colors:", colors)

    return central_wavelengths, colors
