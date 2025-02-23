import matplotlib
matplotlib.use('TkAgg')  # You can replace 'Qt5Agg' with other backends like 'TkAgg' or 'GTK3Agg'
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
from tool_defs import *


# ---------------------------- wavelenghts and materials --------------------------
# Kofman 2019, n ice in uvis range: 200-757 nm
ice_uvis_data = pd.read_csv('../data/opt_cons/kofman_uvis_file.txt', sep=" ", header=21)
ice_uvis_data = ice_uvis_data.drop(ice_uvis_data.columns[-1], axis=1)
names = ['lambda', 'Rl', 'e_Rl', 'nl-10K', 'nl-30K', 'nl-50K', 'nl-70K', 'nl-90K', 'nl-110K', 'nl-130K', 'nl-150K']
names = ['lambda', 'Rl', 'e_Rl', 'nl-10K', 'nl-30K', 'nl-50K', 'nl-70K', 'nl-90K', 'nl-110K', 'nl-130K', 'n']
ice_uvis_data.columns = names

# Mastrapa 2008, n ice in nir: 1.1 - 2.6 micron
ice_nir_data = pd.read_csv('../data/opt_cons/nk/n_k/crys_90K.txt', sep="\t", header=5)
ice_nir_data.columns = ['lambda', 'n', 'k', 'T']
ice_nir_data['lambda'] = ice_nir_data['lambda']*1000    # convert to nm

# combine ice data:
# Combine UVIS and NIR data into one DataFrame
ice_data_comb = pd.concat([ice_uvis_data[['lambda', 'n']], ice_nir_data[['lambda', 'n']]], ignore_index=True)
# Sort the DataFrame by 'lambda' column
ice_data_comb = ice_data_comb.sort_values(by='lambda').reset_index(drop=True)

# Warren 2008: n-ice UV to IR
ice_data_warren = pd.read_csv('../data/opt_cons/warren_2008_ASCIItable.txt', sep='\s+', header=None)
ice_data_warren.columns =['lambda', 'n', 'k']
ice_data_warren['lambda']= ice_data_warren['lambda']*1000     # convert to nm
# print(ice_data)

# Segelstein: n liquid water  10nm - >10 micron
water_data = pd.read_csv('../data/opt_cons/Segelstein.txt', sep="\t", header=2)
water_data.columns = ["lambda", "n", "k"]
water_data['lambda'] = water_data['lambda']*1000        # convert to nm

plt.figure()
plt.plot(ice_data_warren['lambda'], ice_data_warren['n'], label="Ice: Warren(2008)")
# plt.plot(ice_uvis_data['lambda'], ice_uvis_data['n'], label="Ice: Kofman(2019)")
# plt.plot(ice_nir_data['lambda'], ice_nir_data['n'], label="Ice: Mastrapa(2008)")
plt.plot(water_data['lambda'], water_data["n"], label="Water: Segelstein(1981)")
plt.xlabel("wavelength [nm]")
plt.xlim(200, 5000)
plt.ylim(0.7, 2)
plt.ylabel("n")
plt.legend()
plt.show()

# wavelength selection and quick color matches for optical phenomena
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
    else:
        # If input_str is not predefined, return empty lists
        central_wavelengths = []
        colors = []

    print("Selected wavelengths [nm]:", central_wavelengths)
    print("Colors:", colors)

    return central_wavelengths, colors


# ----------------------------- cone projections -------------------------------------

def plot_spice_scene_animated(spectrum_str, ice_model_str, flyby_name, hours_from_closest_approach, orbit_steps,
                              frame_steps, instrument_name, plot_rainbow=True, plot_halo=True, plot_corona=True):
    """
        Plots an animated 3D scene of a spacecraft's flyby around Enceladus, including visualizations
        of scattering phenomena such as rainbows, halos, and coronae, based on the spacecraft's instrument
        data and selected scattering models.

        Parameters:
        - spectrum_str (str): The name of the spectrum to be used, influencing the color and wavelength data.
        - ice_model_str (str): The type of ice model to be used for scattering calculations (e.g., "warren", "combined").
        - flyby_name (str): The name of the flyby for which the data is being plotted.
        - hours_from_closest_approach (float): The time range (in hours) from the closest approach to be considered.
        - orbit_steps (int): The number of steps to divide the orbit into for animation.
        - frame_steps (int): The time step (in seconds) for each frame of the animation.
        - instrument_name (str): The name of the instrument being used to gather data during the flyby.
        - plot_rainbow (bool): Whether to include a visualization of rainbows in the plot (default is True).
        - plot_halo (bool): Whether to include a visualization of halos in the plot (default is True).
        - plot_corona (bool): Whether to include a visualization of coronae in the plot (default is True).

        Returns:
        - None: This function produces a 3D plot visualizing the flyby scene and scattering effects.

        Notes:
        - The function requires the SPICE toolkit for trajectory and position calculations.
        - Various visual aspects of the scene, such as the Sun, Enceladus, and scattering effects, are created using Plotly's graphical capabilities.
        """
    # ------------------------------------------------ SPICE ----------------------------------------------
    # projection tool is based in the Enceladus centered reference frame
    # use casini and sun vectors observed from enceladus
    # in the J2000 frame to minimize movement of the sun -> assume sun position is constant
    print("------------------ SPICE -----------------------")
    METAKR = "../data/cassMetaK.txt"
    SCLKID = -82
    spice.furnsh(METAKR)

    # select time range and fly-by
    utc_range = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach)
    utc_start = utc_range[0]
    utc_end = utc_range[1]
    print("start UTC = ", utc_start)
    print("end UTC = ", utc_end)
    et_start = spice.str2et(utc_start)
    et_end = spice.str2et(utc_end)
    print("ET start seconds past J2: {} ".format(et_start))
    print("ET end seconds past J2: {} ".format(et_end))

    # time steps
    step = orbit_steps
    times = np.array([x * (et_end - et_start) / step + et_start for x in range(step)])

    # position of the sun as seen from Enceladus in the J2000 reference frame(km)
    pos_sun, ltime = spice.spkpos('SUN', times, 'J2000', 'LT+S', 'ENCELADUS')
    # Position of CASSINI as seen from Enceladus in the J2000 frame (km)
    pos_cas, ltime_cas = spice.spkpos('CASSINI', times, 'J2000', 'NONE', 'ENCELADUS')

    sc_cord = pos_cas
    if sc_cord.ndim == 1:
        # Compute the norm directly
        max_sc_distance = np.linalg.norm(sc_cord)
    else:
        sc_norms = np.linalg.norm(sc_cord, axis=1)  # Compute the norm of each row
        max_sc_distance = np.max(sc_norms)          # Find the maximum norm
    cs_radius = max_sc_distance                     # set max distance to radius of celestial sphere

    # get mean SHT orientation in the J2000 frame from Enceladus spin-axis in IAU Enceladus frame
    mean_time = np.mean(times)
    pform2 = spice.pxform("IAU_ENCELADUS", 'J2000', float(mean_time))  # get transformation matrix from IAU_E to J200
    plume_axis = np.array([0, 0, -1])  # orientation of SHT in IAU_enceladus
    plume_axis_j2000 = list(spice.mxv(pform2, plume_axis))

    # filter times for those available in the spice camera kernel
    frame_steps = frame_steps           # size of the steps in seconds
    times_filtered = get_remaining_ets(flyby_name, frame_steps, start_et=et_start, end_et=et_end)
    times_filtered_utc = spice.et2utc(times_filtered, "C", 0)

    # get all cassini locations, instrument coord systems and fovs for filtered times
    instrument_times = np.array(times_filtered)
    # get fov
    room = 4  # the maximum number of 3-dimensional vectors that can be returned in `bounds'.
    [shape, insfrm, bsight, n, bounds] = spice.getfvn(instrument_name, room)
    bounds = np.vstack((bounds, [0, 0, 0]))  # add the origin of the fov (tip of the pyramid)
    fov_list = []                            # create an empty list to store the fov vertices for each et
    cord_sys_list = []                       # create an empyt lists which stores the coordinate system axes
    for ins_time in instrument_times:
        iss_point, ltime_iss = spice.spkpos('CASSINI', ins_time, 'J2000', 'NONE', 'ENCELADUS')
        pform = spice.pxform(instrument_name, 'J2000', ins_time)      # rotation matrix for specific et

        # convert Instrument frame bounds to J2000 frame
        bounds_j2000 = np.zeros((5, 3))
        for i in range(len(bounds)):
            bounds_j2000[i] = spice.mxv(pform, bounds[i])

        # needs to be size of cs_sphere
        iss_fov_vertices = bounds_j2000 * cs_radius + iss_point
        fov_list.append(iss_fov_vertices)

        # get cassini iss coordinate system directions in the J2000 frame
        vector_length = 0.1 * cs_radius
        iss_x_axis = spice.mxv(pform, np.array([1, 0, 0])) * vector_length + iss_point
        iss_y_axis = spice.mxv(pform, np.array([0, 1, 0])) * vector_length + iss_point
        iss_z_axis = spice.mxv(pform, np.array([0, 0, 1])) * vector_length + iss_point
        cord_array = np.vstack((iss_x_axis, iss_y_axis, iss_z_axis))
        cord_sys_list.append(cord_array)

    # get average of sun positions
    mean_sun_cord = np.mean(pos_sun, axis=0)
    print("mean sun coordinate", mean_sun_cord)
    pos_max = pos_sun[-1] - pos_sun[0]
    print("maximum difference in lighting geometry [deg]", angle_between_vectors(pos_sun[-1], pos_sun[0])/2)

    # position of Saturn as seen from Enceladus in the J2000 reference frame (km) for each filtered time
    pos_sat , ltime_sat =  spice.spkpos("SATURN", instrument_times, 'J2000', 'None', 'ENCELADUS')
    # get velocity vector of enceladus as seen from Saturn to indicate e-ring direction
    state_enc, ltimev = spice.spkezr("ENCELADUS", instrument_times, "J2000", "LT+S", "SATURN")
    # Extract the last three values from each array normalize and create length equal to celestial sphere
    enc_vel = [state[-3:] / spice.vnorm(state[-3:]) * cs_radius for state in state_enc]


    # get array with distances
    sc_target_distance = np.linalg.norm(pos_cas, axis=1)
    closest_distance = np.min(sc_target_distance)
    closest_distance_index = np.argmin(sc_target_distance)

    closest_time_et = times[closest_distance_index]
    closest_time_utc = spice.et2utc(closest_time_et, 'C', 3)
    print('time of closest distance', closest_time_utc)
    print("closest target distance", closest_distance)
    print("closest target altitude", closest_distance - 252.1)
    # Clean up the kernels
    spice.kclear()

    # get phase angles
    phase_angles = []
    for i in range(len(pos_sun)):
        a = angle_between_vectors(pos_sun[i], pos_cas[i])
        phase_angles.append(a)

    print("phase angle -1h", phase_angles[0])
    print("phase angle +1h", phase_angles[-1])
    plt.figure()
    plt.title("Phase angles during fly-by")
    plt.xlabel("ET")
    plt.ylabel("phase angle")
    plt.plot(times, phase_angles)
    plt.axvline(closest_time_et, label="closest approach", color="black")
    plt.legend()
    plt.show()

    # ----- create bodies ------
    # Create traces for each body
    traces = []

    # Enceladus
    enceladus_radius = 252.1
    xe, ye, ze = sphere(enceladus_radius)
    trace_enceladus = go.Surface(z=ze, x=xe, y=ye, colorscale=[[0, 'white'], [1, 'white']], showscale=False,
                                 name="Enceladus")
    traces.append(trace_enceladus)

    # Plume
    xp, yp, zp = cone(600, 20, direction_vector=[0, 0, 0], y_rotate=90, z_shift=-200)
    xp, yp, zp = cone(800, 10, direction_vector=plume_axis_j2000)
    trace_plume = go.Surface(z=zp, x=xp, y=yp, colorscale=[[0, 'grey'], [1, 'grey']], opacity=0.3, showscale=False,
                             name="Plume")
    traces.append(trace_plume)

    # Plume observation point
    plume_obs_alt = 40
    plume_scatter_point = np.array(plume_axis_j2000) * (plume_obs_alt+ enceladus_radius)
    trace_observation_point = go.Scatter3d(x=[plume_scatter_point[0]], y=[plume_scatter_point[1]],
                                           z=[plume_scatter_point[2]], mode='markers',
                                           marker=dict(size=3, color='grey'), name='Scatter point')
    traces.append(trace_observation_point)

    # Celestial sphere around plume observation
    xcs, ycs, zcs = sphere(cs_radius, x_shift=plume_scatter_point[0], y_shift=plume_scatter_point[1],
                           z_shift=plume_scatter_point[2])
    trace_celestial_sphere = go.Surface(z=zcs, x=xcs, y=ycs, colorscale=[[0, 'grey'], [1, 'grey']], opacity=0.05,
                                        showscale=False, name="Celestial sphere",  showlegend=True)
    traces.append(trace_celestial_sphere)

    # Sun
    sun_cord = mean_sun_cord
    # transform vector to declination and right ascension on celestial sphere
    projection_xy = np.array([sun_cord[0], sun_cord[1], 0])
    sun_declination = angle_between_vectors(projection_xy, np.array(sun_cord))
    # adjust sign
    if sun_cord[2] <= 0:
        sun_declination = -sun_declination

    sun_right_ascension = angle_between_vectors(np.array([1,0,0]),projection_xy)
    if sun_cord[1] <= 0:
        sun_right_ascension = - sun_right_ascension

    sun_lat_rad, sun_lon_rad = sun_declination*np.pi/180, sun_right_ascension*np.pi/180
    rotation_matrix_lat = np.array([[np.cos(-sun_lat_rad), 0, np.sin(-sun_lat_rad)],
                                    [0, 1, 0],
                                    [-np.sin(-sun_lat_rad), 0, np.cos(-sun_lat_rad)]])

    rotation_matrix_lon = np.array([[np.cos(sun_lon_rad), -np.sin(sun_lon_rad), 0],
                                [np.sin(sun_lon_rad), np.cos(sun_lon_rad), 0],
                                [0, 0, 1]])

    # define sun
    sun_cord_unit = sun_cord/np.linalg.norm(sun_cord)
    sun_cord_short = sun_cord_unit*cs_radius*1.2
    # Perform the rotations
    sun_cord_lat = np.dot(rotation_matrix_lat, np.array([1,0,0])*cs_radius*1.2)
    sun_cord_lat_lon = np.dot(rotation_matrix_lon, sun_cord_lat)
    xs, ys, zs = sun_cord_lat_lon

    # print(xs, ys, zs)
    trace_sun = go.Scatter3d(x=[xs], y=[ys], z=[zs], mode='markers',
                             marker=dict(size=6, color='gold'), name='Sun')
    # traces.append(trace_sun)
    trace_sun_check = go.Scatter3d(x=[sun_cord_short[0]], y=[sun_cord_short[1]], z=[sun_cord_short[2]], mode='markers',
                             marker=dict(size=6, color='yellow'), name='Sun check')
    # traces.append(trace_sun_check)
    # Define the line between sub-solar point and Enceladus center
    x_start, y_start, z_start = xs, ys, zs # Starting point
    x_end, y_end, z_end = 0, 0, 0  # Ending point
    sun_vector = np.array([x_end-x_start, y_end-y_start, z_end-z_start])

    # Create a trace for the line
    line_trace_sun = go.Scatter3d(
        x=[x_start, x_end],  # X-coordinates of the line's start and end points
        y=[y_start, y_end],  # Y-coordinates of the line's start and end points
        z=[z_start, z_end],  # Z-coordinates of the line's start and end points
        mode='lines',  # Specify mode as lines
        line=dict(color='gold', width=3),  # Specify line color and width
        name='Sun-Enc vector'  # Name of the trace
    )
    traces.append(line_trace_sun)

    # plot spacecraft trajectory
    sc_cordt = sc_cord.T
    if sc_cord.ndim == 1:
        trace_sc = go.Scatter3d(x=[sc_cordt[0]], y=[sc_cordt[1]], z=[sc_cordt[2]], mode='markers',
                     marker=dict(size=3, color='black'), name='Cassini position')
    else:
        trace_sc = go.Scatter3d(x=sc_cordt[0], y=sc_cordt[1], z=sc_cordt[2], mode='lines', line=dict(color='black', width=3),
                                name="Cassini trajectory")
        # trace_sc_rotated = go.Scatter3d(x=rotated_sc[0], y=rotated_sc[1], z=rotated_sc[2], mode='lines',
        #                         line=dict(color='green', width=3),
        #                         name="Cassini trajectory rotated")
    traces.append(trace_sc)

    # select refractive indices model
    if ice_model_str == "warren":
        ice_data = ice_data_warren
    elif ice_model_str == "combined":
        ice_data = ice_data_comb
    # select wavelengths and data
    central_wavelengths, colors = get_wavelengths_and_colors(spectrum_str)

    if len(central_wavelengths) == 1:
        opacity =0.5
    else:
        opacity = 1

    # select data for specific wavelengths
    df_water = select_lambdas(water_data, central_wavelengths)
    df_ice = select_lambdas(ice_data, central_wavelengths)

    lam_water = df_water['lambda']
    n_water = df_water['n']

    lam_ice = df_ice['lambda']
    n_ice = df_ice['n']
    ni = 1

    # Rainbow
    if plot_rainbow:
        # rainbow
        n_rainbow = n_water/ni
        k= 1
        rainbow_incidence = np.arccos(np.sqrt((n_rainbow**2-1)/(k*(2+k))))
        rainbow_refract = np.arcsin(np.sin(rainbow_incidence)/n_rainbow)
        rainbow_scat = (2 * rainbow_incidence - 2*(1+k)*rainbow_refract) * 180/np.pi
        rainbow_scat = list(rainbow_scat)
        print("rainbow scattering angles [deg]:", rainbow_scat)

        for i in range(len(rainbow_scat)):
            xr, yr, zr = cone(cs_radius, -1*rainbow_scat[i], [0,0,0], y_rotate=-sun_declination,
                              z_rotate=sun_right_ascension, x_shift=plume_scatter_point[0],
                              y_shift=plume_scatter_point[1], z_shift=plume_scatter_point[2])
            trace = go.Surface(z=zr, x=xr, y=yr, name=f'Rainbow:{int(list(lam_water)[i])}nm',
                               colorscale=[[0, colors[i]], [1, colors[i]]], opacity=opacity, showscale=False,
                               showlegend=True)
            traces.append(trace)
        # add boundary range for vims check
        if spectrum_str == "vims_check":
            rainrange =2
            rainbow_scat_bounds = [rainbow_scat[0]+rainrange, rainbow_scat[1]-rainrange]
            print("rainbow scattering angles boundaries[deg]:", rainbow_scat_bounds)
            bound_color = ["violet",  "brown"]
            for i in range(len(rainbow_scat_bounds)):
                xr, yr, zr = cone(cs_radius, -1 * rainbow_scat_bounds[i], [0, 0, 0], y_rotate=-sun_declination,
                                  z_rotate=sun_right_ascension, x_shift=plume_scatter_point[0],
                                  y_shift=plume_scatter_point[1], z_shift=plume_scatter_point[2])
                trace = go.Surface(z=zr, x=xr, y=yr, name='Rainbow boundary',
                                   colorscale=[[0, bound_color[i]], [1, bound_color[i]]], opacity=1, showscale=False,
                                   showlegend=True)
                traces.append(trace)
    # Halo
    if plot_halo:
        n_halo = n_ice/ni
        refracting_angle = 60 * np.pi/180
        d_min = 2 * np.arcsin(n_halo/ni*np.sin(refracting_angle/2)) - refracting_angle
        d_min_deg = list(d_min * 180/np.pi)
        print("halo scattering angles [deg]", d_min_deg)

        for i in range(len(d_min_deg)):
            xh, yh, zh = cone(cs_radius, d_min_deg[i], y_rotate=180-sun_declination, z_rotate=sun_right_ascension,
                              x_shift=plume_scatter_point[0], y_shift=plume_scatter_point[1],
                              z_shift=plume_scatter_point[2])
            trace = go.Surface(z=zh, x=xh, y=yh, name=f'Halo:{int(list(lam_ice)[i])}nm',
                               colorscale=[[0, colors[i]], [1, colors[i]]], opacity=opacity, showscale=False, showlegend=True)
            traces.append(trace)
        if spectrum_str == "vims_check":
            halorange =1
            d_min_deg_bounds = [d_min_deg[0]+halorange, d_min_deg[1]-halorange]
            print("halo scattering angles boundaries [deg]", d_min_deg_bounds)
            bound_color = ["violet",  "brown"]
            for i in range(len(d_min_deg)):
                xh, yh, zh = cone(cs_radius, d_min_deg_bounds[i], y_rotate=180 - sun_declination, z_rotate=sun_right_ascension,
                                  x_shift=plume_scatter_point[0], y_shift=plume_scatter_point[1],
                                  z_shift=plume_scatter_point[2])
                trace = go.Surface(z=zh, x=xh, y=yh, name=f'Halo Boundary',
                                   colorscale=[[0, bound_color[i]], [1, bound_color[i]]], opacity=1, showscale=False,
                                   showlegend=True)
                traces.append(trace)

    # Corona
    if plot_corona:
        if spectrum_str == "vims_check":
            scattering_angles = [1 , 20]
            print("corona scattering angles range for:",
                  scattering_angles , "degrees")
            bound_color = ["purple",  "red"]
            for i in range(len(scattering_angles)):

                xc, yc, zc = cone(cs_radius, scattering_angles[i], y_rotate=180 - sun_declination,
                                  z_rotate=sun_right_ascension, x_shift=plume_scatter_point[0],
                                  y_shift=plume_scatter_point[1], z_shift=plume_scatter_point[2])
                trace = go.Surface(z=zc, x=xc, y=yc, name='Corona boundary',
                                   colorscale=[[0, bound_color[i]], [1, bound_color[i]]], opacity=opacity, showscale=False,
                                   showlegend=True, legendgroup="Corona")
                traces.append(trace)
        else:
            cor_angle_range = 10
            corona_scat = np.linspace(0, cor_angle_range, cor_angle_range*1000)*np.pi/180     # scattering angle range
            wavelength = lam_ice*1e-9
            wavelength = lam_ice*1e-9
            r = 10e-6                                                    # particle radius
            x = list(2 * np.pi * r / wavelength)                        # size parameter
            wavelength = list(wavelength)
            plt.figure()
            for j in range(len(x)):
                bes_argument = x[j] * np.sin(corona_scat)
                j1 = scp.jv(1, bes_argument)        # Using scipy.special.jv for Bessel function of the first kind
                io_corona = (x[j]*(1+np.cos(corona_scat))/2* j1/bes_argument)**2
                # Plot the Bessel function
                plt.plot(corona_scat * 180 / np.pi, io_corona, color=colors[j], label=f"{int(wavelength[j]*1e9)}nm")
                # Find peaks in the data
                peaks, _ = sc.signal.find_peaks(io_corona)
                # Select the scattering angles of the first 3 peaks
                number_of_peaks = 2
                scattering_angles = corona_scat[peaks[:number_of_peaks]]
                print(f"corona scattering angles of the first 3 maxima for {wavelength[j]}:",
                      scattering_angles * 180 / np.pi, "degrees")
                for angle in scattering_angles:
                    xc, yc, zc = cone(cs_radius, angle * 180 / np.pi,y_rotate=180-sun_declination,
                                      z_rotate=sun_right_ascension , x_shift=plume_scatter_point[0],
                                      y_shift=plume_scatter_point[1], z_shift=plume_scatter_point[2])
                    trace = go.Surface(z=zc, x=xc, y=yc, name=f'Corona:{int(wavelength[j]*1e9)}nm',
                                       colorscale=[[0, colors[j]], [1, colors[j]]], opacity=1, showscale=False, showlegend=True, legendgroup="Corona")
                    traces.append(trace)

            # plt.title('Corona Intensity Function')
            plt.xlabel(r'$\theta$ [$^\circ$]')
            plt.ylabel('Intensity ')
            plt.yscale('log')
            plt.ylim(10**(-6), 10**(4))
            plt.grid(True)
            plt.legend()
            plt.show()


    # get list of traces for each frame
    frames_traces = []
    # fov traces
    for i in range(len(times_filtered)):
        traces_basic = copy.deepcopy(traces)  # define the basic traces frame without fov

        # single frame fov
        v = fov_list[i]
        # Generate list of sides' polygons of the pyramid
        faces = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
                 [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

        # Extract x, y, z coordinates of each vertex
        x, y, z = v[:, 0], v[:, 1], v[:, 2]


        # Define traces for vertices and faces
        # Saturn position for each frame
        # define saturn cord
        sat_cord_unit = pos_sat[i] / np.linalg.norm(pos_sat[i])
        sat_cord_short = sat_cord_unit * cs_radius * 1
        trace_sat = go.Scatter3d(x=[sat_cord_short[0]], y=[sat_cord_short[1]], z=[sat_cord_short[2]], mode='markers',
                                 marker=dict(size=8, color='tan'), name='Saturn')
        # add saturn trace
        # traces_basic.append(trace_sat)
        # Cassini position in each frame
        trace_tip = go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers',
                                marker=dict(size=3, color='black'), name='Cassini position')
        traces_basic.append(trace_tip)
        for face in faces:
            x_face = [vertex[0] for vertex in face]
            y_face = [vertex[1] for vertex in face]
            z_face = [vertex[2] for vertex in face]
            trace_face = go.Mesh3d(x=x_face, y=y_face, z=z_face, color='cyan', opacity=0.25, showlegend=False)
            traces_basic.append(trace_face)

        # plot ISS cordinate system frame
        system_vectors = cord_sys_list[i]
        iss_x_axis = system_vectors[0]
        iss_y_axis = system_vectors[1]
        iss_z_axis = system_vectors[2]
        trace_vector_iss_x = go.Scatter3d(x=[x[-1], iss_x_axis[0]], y=[y[-1], iss_x_axis[1]],
                                          z=[z[-1], iss_x_axis[2]],
                                          mode='lines',
                                          # marker=dict(size=10, symbol='cone', color='blue'),
                                          line=dict(color='lightcyan', width=3),
                                          name='iss-x vector')
        traces_basic.append(trace_vector_iss_x)
        trace_vector_iss_y = go.Scatter3d(x=[x[-1], iss_y_axis[0]], y=[y[-1], iss_y_axis[1]],
                                          z=[z[-1], iss_y_axis[2]],
                                          mode='lines',
                                          # marker=dict(size=10, symbol='cone', color='blue'),
                                          line=dict(color='darkcyan', width=3),
                                          name='iss-y vector')
        traces_basic.append(trace_vector_iss_y)
        trace_vector_iss_z = go.Scatter3d(x=[x[-1], iss_z_axis[0]], y=[y[-1], iss_z_axis[1]],
                                          z=[z[-1], iss_z_axis[2]],
                                          mode='lines',
                                          # marker=dict(size=10, symbol='cone', color='blue'),
                                          line=dict(color='cyan', width=3),
                                          name='iss-z vector')
        traces_basic.append(trace_vector_iss_z)

        frames_traces.append(traces_basic)

    print("frame_traces list shape (#frames, #traces/frame)", np.shape(frames_traces))
    # Create figure
    fig = go.Figure()

    for trace in frames_traces[0]:
        fig.add_trace(trace)

    # Define frames
    frames = []

    # Create frames with traces for each frame
    for i, traces in enumerate(frames_traces):
        frame = go.Frame(
            data=traces,
            name=f'frame_{i}',
            traces=list(range(len(traces)))  # Each trace corresponds to a different frame
        )
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Define slider steps
    slider_steps = []
    for i in range(len(frames)):
        step = {"args": [
            [f"frame_{i}"],
            {"frame": {"duration": 100, "redraw": True}, "mode": "immediate", "transition": {"duration": 100}}
        ],
            "label": times_filtered_utc[i],
            "method": "animate"}
        slider_steps.append(step)

    # Update layout and define buttons
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-0.5, y=-1, z=0.1))

    fig.update_layout(title='Enceladus fly-by {}'.format(flyby_name),

        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True,
                                        "transition": {"duration": 100}}],
                        "label": "Play",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 137},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"t": 50},
            "steps": slider_steps
        }], scene_camera=camera
    )
    fig.write_html("index.html")

    # Show figure
    fig.show()



if __name__ == "__main__":

    # plot projections and trajectory
    plot_spice_scene_animated(spectrum_str="vis", ice_model_str='warren', flyby_name="E11", hours_from_closest_approach=0.5,
                              orbit_steps=400, frame_steps=120, instrument_name='CASSINI_ISS_WAC',
                              plot_rainbow=True, plot_halo=False, plot_corona=False)


