import matplotlib
matplotlib.use('TkAgg')  # You can replace 'Qt5Agg' with other backends like 'TkAgg' or 'GTK3Agg'
from pyvims import VIMS
from matplotlib.path import Path
from matplotlib.colors import Normalize
from tool_defs import *


def plot_spice_scene_ring(flyby_name, hours_from_closest_approach, orbit_steps, frame_steps, instrument_name,
                           vims_cubes=None , i1=254, d_sight=None):
    """
    Plots an interactive 3D scene of celestial bodies and spacecraft trajectories during a Cassini flyby.

    original update to plot_spice_scene_animated from projection_tool_enc_centered.py: the center is now Saturn.
    plot a single vims cube for each time step with a slider if vims_cubes are given as input.
    If not acts as regular plot_spice_scene_animated based on input time range.

    Parameters:
    -----------
    flyby_name : str
        The name of the flyby event to visualize.
    hours_from_closest_approach : float
        The number of hours from the closest approach to retrieve the time range for the flyby.
    orbit_steps : int
        The number of steps to divide the orbit time range for plotting.
    frame_steps : int
        The number of steps in seconds between frames for time filtering.
    instrument_name : str
        The name of the instrument to retrieve field of view (FOV) information from SPICE kernels.
    vims_cubes : list of VIMSCube, optional
        A list of VIMS cube objects to plot. If provided, a single cube will be plotted for each time step.
    i1 : int, optional
        The index of the VIMS-IR channel  (default is 254).
    d_sight : float, optional
        The distance to sight for celestial sphere rendering. If not provided, projection at Cassini's position.

    Returns:
    --------
    None
        The function displays the plot directly.

    """
    # ------------------------------------------------ SPICE ----------------------------------------------
    print("------------------ SPICE -----------------------")
    METAKR = "../data/cassMetaK.txt"
    SCLKID = -82
    spice.furnsh(METAKR)

    if vims_cubes:
        times = []
        time_utc = []
        cube_names = []
        for vcube in vims_cubes:
            cube_name = vcube.img_id
            cube_names.append(cube_name)
            # get et time
            time_et_i = vcube.et_start + (vcube.et_stop-vcube.et_start)/2
            # append times to lists
            time_utc.append(spice.et2utc(time_et_i, "c", 0))
            times.append(time_et_i)
        times_filtered = times
        times_filtered_utc = time_utc
        print(times_filtered_utc)

    else:
        # select time range and fly-by
        utc_range = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach)
        utc_range_end = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach/2)
        utc_start = utc_range[0]
        utc_end = utc_range_end[1]
        print("start UTC = ", utc_start)
        print("end UTC = ", utc_end)
        et_start = spice.str2et(utc_start)
        et_end = spice.str2et(utc_end)
        print("ET start seconds past J2: {} ".format(et_start))
        print("ET end seconds past J2: {} ".format(et_end))

        # time steps
        step = orbit_steps
        times = np.array([x * (et_end - et_start) / step + et_start for x in range(step)])

        # filter times for those available in the spice camera kernel
        frame_steps = frame_steps  # size of the steps in seconds
        times_filtered = get_remaining_ets(flyby_name, frame_steps, start_et=et_start, end_et=et_end)
        times_filtered_utc = spice.et2utc(times_filtered, "C", 0)

    # position of the sun as seen from Enceladus in the J2000 reference frame(km)
    pos_sun, ltime = spice.spkpos('SUN', times, 'J2000', 'LT+S', 'SATURN')
    # Position of CASSINI as seen from Enceladus in the J2000 frame (km)
    pos_cas, ltime_cas = spice.spkpos('CASSINI', times, 'J2000', 'NONE', 'SATURN')


    sc_cord = pos_cas
    if sc_cord.ndim == 1:
        # Compute the norm directly
        max_sc_distance = np.linalg.norm(sc_cord)
    else:
        sc_norms = np.linalg.norm(sc_cord, axis=1)  # Compute the norm of each row
        max_sc_distance = np.max(sc_norms)          # Find the maximum norm
    if d_sight:
        cs_radius = d_sight
    else:
        cs_radius = max_sc_distance                     # set max distance to radius of celestial sphere
        # Celestial sphere around plume observation
        cs_radius = 1.2 * np.linalg.norm(pos_cas[0])

    # get mean SHT orientation in the J2000 frame from Enceladus spin-axis in IAU Enceladus frame
    mean_time = np.mean(times)
    pform2 = spice.pxform("IAU_ENCELADUS", 'J2000', float(mean_time))  # get transformation matrix from IAU_E to J200
    plume_axis = np.array([0, 0, -1])  # orientation of SHT in IAU_enceladus
    plume_axis_j2000 = list(spice.mxv(pform2, plume_axis))

    # get all cassini locations, instrument coord systems and fovs for filtered times
    instrument_times = np.array(times_filtered)
    # get fov
    room = 4  # the maximum number of 3-dimensional vectors that can be returned in `bounds'.
    [shape, insfrm, bsight, n, bounds] = spice.getfvn(instrument_name, room)
    bounds = np.vstack((bounds, [0, 0, 0]))  # add the origin of the fov (tip of the pyramid)
    fov_list = []                            # create an empty list to store the fov vertices for each et
    cord_sys_list = []                       # create an empyt lists which stores the coordinate system axes
    bsights = []
    for ins_time in instrument_times:
        iss_point, ltime_iss = spice.spkpos('CASSINI', ins_time, 'J2000', 'NONE', 'SATURN')
        pform = spice.pxform(instrument_name, 'J2000', ins_time)      # rotation matrix for specific et
        if d_sight:
            bsight_i = spice.mxv(pform, bsight)
            bsights.append(bsight_i)

        # convert Instrument frame bounds to J2000 frame
        bounds_j2000 = np.zeros((5, 3))
        for i in range(len(bounds)):
            bounds_j2000[i] = spice.mxv(pform, bounds[i])

        # needs to be size of cs_sphere
        iss_fov_vertices = bounds_j2000 * cs_radius + iss_point
        fov_list.append(iss_fov_vertices)

        # get cassini iss coordinate system directions in the J2000 frame
        if d_sight:
            vector_length = 0.01 * cs_radius
        else:
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

    # position of Dione
    pos_dio , ltime_dio =  spice.spkpos("DIONE", instrument_times, 'J2000', 'None', 'SATURN')

    state_dio , ltime_dio =  spice.spkezr("DIONE", instrument_times, 'J2000', 'None', 'SATURN')
    pos_dio_sat = [state[:3] for state in state_dio]
    vel_dio = [state[-3:] for state in state_dio]

    # position ot tethys
    state_tet, ltime_tet = spice.spkezr("TETHYS", instrument_times, 'J2000', 'None', 'SATURN')
    pos_tet = [state[:3] for state in state_tet]
    vel_tet = [state[-3:] for state in state_tet]

    # position phoebe
    days_range = 300
    pho_start = times[0] - 24 * 60 * 60 * days_range
    pho_end = times[-1] + 24 * 60 * 60 * days_range
    pho_times = np.linspace(pho_start, pho_end, 1000)
    state_pho, _ = spice.spkezr("PHOEBE", pho_times, 'J2000', 'None', 'SATURN')
    pos_pho = [state[:3] for state in state_pho]
    pos_pho2, _ = spice.spkpos("PHOEBE", instrument_times, 'J2000', 'None', 'SATURN')
    #S2004 S52
    days_range = 600
    pho_start = times[0] - 24 * 60 * 60 * days_range
    pho_end = times[-1] + 24 * 60 * 60 * days_range
    pho_times = np.linspace(pho_start, pho_end, 1000)
    # pos_gri, _ = spice.spkpos("65152", pho_times, 'J2000', 'None', 'SATURN')
    # pos_irr, _ = spice.spkpos("65152", pho_times, 'J2000', 'None', 'SATURN')

    # rocks
    pos_prom, ltime_prom = spice.spkpos("PROMETHEUS", instrument_times, 'J2000', 'None', 'SATURN')
    pos_hel, ltime_hel = spice.spkpos("HELENE", instrument_times, 'J2000', 'None', 'SATURN')
    pos_tel, ltime_tel = spice.spkpos("TELESTO", instrument_times, 'J2000', 'None', 'SATURN')
    pos_cal, ltime_cal = spice.spkpos("CALYPSO", instrument_times, 'J2000', 'None', 'SATURN')

    pos_pol, ltime_pol = spice.spkpos("POLYDEUCES", instrument_times, 'J2000', 'None', 'SATURN')

    # position of cassini but at filtered time steps
    pos_cas_fil, ltime_cas_fil = spice.spkpos('CASSINI', instrument_times, 'J2000', 'NONE', 'SATURN')

    # state of enceladus a sseen from cassini
    state_enc, ltimev = spice.spkezr("ENCELADUS", instrument_times, "J2000", "LT+S", "SATURN")
    # Extract the last three values from each array
    enc_vel = [state[-3:]/spice.vnorm(state[-3:]) for state in state_enc]
    enc_pos = [state[:3] for state in state_enc]

    state_cas, _ = spice.spkezr("CASSINI", instrument_times, "J2000", "LT+S", "SATURN")
    cas_vel = [state[-3:]/spice.vnorm(state[-3:]) for state in state_cas]

    # need spherical coordinates radial velocty and tangential velocity? spinning of saturn does not matter for rel v so can direclty compute it
    state_enc_cas, _ = spice.spkezr("ENCELADUS", instrument_times, "J2000", "LT+S", "CASSINI")

    # get array with distances
    # sc_target_distance = np.linalg.norm(pos_cas, axis=1)
    # closest_distance = np.min(sc_target_distance)
    # closest_distance_index = np.argmin(sc_target_distance)
    # closest_time_et = times[closest_distance_index]
    # closest_time_utc = spice.et2utc(closest_time_et, 'C', 3)
    # print('time of closest distance', closest_time_utc)
    # print("closest target distance", closest_distance)
    # print("closest target altitude", closest_distance - 252.1)
    # Clean up the kernels
    spice.kclear()

    # get phase angles
    # phase_angles = []
    # for i in range(len(pos_sun)):
    #     a = angle_between_vectors(pos_sun[i], pos_cas[i])
    #     phase_angles.append(a)
    #
    # print("phase angle -1h", phase_angles[0])
    # print("phase angle +1h", phase_angles[-1])
    # plt.figure()
    # plt.title("Phase angles during fly-by")
    # plt.xlabel("ET")
    # plt.ylabel("phase angle")
    # plt.plot(times, phase_angles)
    # plt.axvline(closest_time_et, label="closest approach", color="black")
    # plt.legend()
    # plt.show()

    # ----- create bodies ------
    # Create traces for each body
    traces = []

    # Saturn
    sat_radius = 58232
    xsat, ysat, zsat = sphere(sat_radius)
    trace_sat = go.Surface(z=zsat, x=xsat, y=ysat, colorscale=[[0, 'tan'], [1, 'tan']], showscale=False,
                                 name="Saturn")
    traces.append(trace_sat)

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
        name='Sun direction'  # Name of the trace
    )
    traces.append(line_trace_sun)

    # get angle of cone cross-section on image plane
    # rotate the cassini positions so that the sun vector coincides with the x-axis
    # this allows for 2d analysis of the cross-sections
    sc_cordt_fil = pos_cas_fil.T
    # make sure rotations happen for the right signs
    if sun_right_ascension > 0:
        rotated_sc = rotate_z(sc_cordt_fil[0], sc_cordt_fil[1], sc_cordt_fil[2], -sun_right_ascension)
    else:
        rotated_sc = rotate_z(sc_cordt_fil[0], sc_cordt_fil[1], sc_cordt_fil[2], sun_right_ascension)

    if sun_declination > 0:
        rotated_sc = rotate_y(rotated_sc[0], rotated_sc[1], rotated_sc[2], sun_declination)
    else:
        rotated_sc = rotate_y(rotated_sc[0], rotated_sc[1], rotated_sc[2], -sun_declination)

    # Phoebe ring
    pho_cordt = np.array(pos_pho).T
    trace_pho = go.Scatter3d(x=pho_cordt[0], y=pho_cordt[1], z=pho_cordt[2], mode='lines',
                             line=dict(color='grey', width=2),
                             name="Phoebe trajectory")
    # traces.append(trace_pho)

    # plot spacecraft trajectory
    sc_cordt = sc_cord.T
    if sc_cord.ndim == 1:
        trace_sc = go.Scatter3d(x=[sc_cordt[0]], y=[sc_cordt[1]], z=[sc_cordt[2]], mode='markers',
                     marker=dict(size=2, color='skyblue'), name='Cassini position')
    else:
        trace_sc = go.Scatter3d(x=sc_cordt[0], y=sc_cordt[1], z=sc_cordt[2], mode='lines', line=dict(color='skyblue', width=2),
                                name="Cassini trajectory")
    traces.append(trace_sc)

    if vims_cubes:
        # VIMS colorscale
        data = np.array([
            cube.data
            for cube in vims_cubes
        ])
        norm = Normalize(vmin=0, vmax=np.percentile(data[:, i1, :, :], 99), clip=True)

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

        # Cassini position in each frame
        trace_tip = go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers',
                                marker=dict(size=3, color='skyblue'), name='Cassini position')
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

        # Enceladus
        trace_enc = go.Scatter3d(x=[enc_pos[i][0]], y=[enc_pos[i][1]], z=[enc_pos[i][2]], mode='markers',
                                 marker=dict(size=4, color='lightgrey'), name='Enceladus')
        traces_basic.append(trace_enc)

        # E-ring
        circle_points = generate_circle_points(np.array([0, 0, 0]), np.linalg.norm(enc_pos[i]), enc_vel[i],
                                               origin=enc_pos[i])
        filtered_circle_points = circle_points
        # Add the filtered circle points to the figure
        ering_trace = go.Scatter3d(x=filtered_circle_points[:, 0], y=filtered_circle_points[:, 1],
                                   z=filtered_circle_points[:, 2],
                                   mode='lines', name='E-ring', line=dict(color='lightgrey', dash='dash'))
        traces_basic.append(ering_trace)

        # Dione
        trace_dio = go.Scatter3d(x=[pos_dio[i][0]], y=[pos_dio[i][1]], z=[pos_dio[i][2]], mode='markers',
                                 marker=dict(size=4, color='darkseagreen'), name='Dione')
        traces_basic.append(trace_dio)

        # dione ring
        dione_ring_radius = np.linalg.norm(pos_dio_sat[i])
        circle_points = generate_circle_points(np.array([0,0,0]),dione_ring_radius, vel_dio[i], origin=pos_dio[i])
        dring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                                   z=circle_points[:, 2],
                                   mode='lines', name='Dione orbit', line=dict(color='darkseagreen', dash='dash', width=2))
        traces_basic.append(dring_trace)

        # Tethys
        trace_tet = go.Scatter3d(x=[pos_tet[i][0]], y=[pos_tet[i][1]], z=[pos_tet[i][2]], mode='markers',
                                 marker=dict(size=4, color='indianred'), name='Tethys')
        traces_basic.append(trace_tet)

        # tethys ring
        tet_ring_radius = np.linalg.norm(pos_tet[i])
        circle_points = generate_circle_points(np.array([0, 0, 0]), tet_ring_radius, vel_tet[i], origin=pos_tet[i])
        tring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                                   z=circle_points[:, 2],
                                   mode='lines', name='Tethys orbit', line=dict(color='indianred', dash='dash'))
        traces_basic.append(tring_trace)

        # Saturn F-ring
        fring_radius = 140220
        circle_points = generate_circle_points(np.array([0, 0, 0]), fring_radius, enc_vel[0], origin=enc_pos[0])
        fring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                                   z=circle_points[:, 2],
                                   mode='lines', name='F-ring', line=dict(color='moccasin'))
        traces_basic.append(fring_trace)
        # Saturn G-ring
        gring_radius = 349554 / 2
        circle_points = generate_circle_points(np.array([0, 0, 0]), gring_radius, enc_vel[0], origin=enc_pos[0])
        gring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                                   z=circle_points[:, 2],
                                   mode='lines', name='G-ring', line=dict(color='peru'))
        traces_basic.append(gring_trace)

        # Saturn's irregular satellites
        trace_prom = go.Scatter3d(x=[pos_prom[i][0]], y=[pos_prom[i][1]], z=[pos_prom[i][2]], mode='markers',
                                 marker=dict(size=3, color='moccasin'), name='Prometheus')
        traces_basic.append(trace_prom)

        trace_hel = go.Scatter3d(x=[pos_hel[i][0]], y=[pos_hel[i][1]], z=[pos_hel[i][2]], mode='markers',
                                 marker=dict(size=3, color='darkseagreen'), name='Helene')
        traces_basic.append(trace_hel)

        trace_pol = go.Scatter3d(x=[pos_pol[i][0]], y=[pos_pol[i][1]], z=[pos_pol[i][2]], mode='markers',
                                 marker=dict(size=3, color='darkseagreen'), name='Polydeuces')
        traces_basic.append(trace_pol)

        trace_tel = go.Scatter3d(x=[pos_tel[i][0]], y=[pos_tel[i][1]], z=[pos_tel[i][2]], mode='markers',
                                 marker=dict(size=3, color='indianred'), name='Telesto')
        traces_basic.append(trace_tel)
        trace_cal = go.Scatter3d(x=[pos_cal[i][0]], y=[pos_cal[i][1]], z=[pos_cal[i][2]], mode='markers',
                                 marker=dict(size=2, color='indianred'), name='Calypso')
        traces_basic.append(trace_cal)

        # ------------------------ VIMS cube show --------------------------
        if vims_cubes:
            cube_instance = [vims_cubes[i]]
            paths = [
                Path(cube.rsky[:, l, s, :].T)
                for cube in cube_instance
                for l in range(cube.NL)
                for s in range(cube.NS)
            ]

            vertices = np.stack([
                path.vertices
                for path in paths
            ])

            if d_sight:
                dist_along_sight = bsights[i] * cs_radius
                dist = np.linalg.norm(dist_along_sight)

                rec_vertices = []
                for vertice in vertices:
                    # a single vertice is an array representing the 4 corner coordinates of a single pixel
                    rec_vertice = np.zeros((4, 3))
                    for r in range(len(vertice)):
                        # convert to rectilinear coordinates
                        x, y, z = spice.radrec(dist, np.deg2rad(vertice[r, 0]), np.deg2rad(vertice[r, 1]))
                        rec_vertice[r] = np.array([x, y, z])
                    # store new rectilinear pixel vertice in list containing all pixels of a single cube
                    rec_vertices.append(rec_vertice)

                # Calculate the centroid of the combined image
                all_rec_vertices = np.array(rec_vertices).reshape(-1, 3)
                centroid = all_rec_vertices.mean(axis=0)

                # shift pixels first to center of the plot (Saturn) and then to Cassini's position
                for rec_ver in rec_vertices:
                    for c in range(len(rec_ver)):
                        rec_ver[c] = rec_ver[c] + pos_cas[i]
            else:
                scale = 10
                dist = np.linalg.norm((pos_cas[i]-enc_pos[i])*scale)
                # dist = np.linalg.norm(pos_dio[0])
                # dist = np.linalg.norm(enc_pos[i]*scale)


                rec_vertices = []
                for vertice in vertices:
                    # a single vertice is an array representing the 4 corner coordinates of a single pixel
                    rec_vertice = np.zeros((4, 3))
                    for r in range(len(vertice)):
                        # convert to rectilinear coordinates
                        x, y, z = spice.radrec(dist, np.deg2rad(vertice[r, 0]), np.deg2rad(vertice[r, 1]))
                        rec_vertice[r] = np.array([x, y, z])
                    # store new rectilinear pixel vertice in list containing all pixels of a single cube
                    rec_vertices.append(rec_vertice)

                # Calculate the centroid of the combined image
                all_rec_vertices = np.array(rec_vertices).reshape(-1, 3)
                centroid = all_rec_vertices.mean(axis=0)

                # shift pixels first to center of the plot (Saturn) and then to Cassini's position
                for rec_ver in rec_vertices:
                    for c in range(len(rec_ver)):
                        rec_ver[c] = rec_ver[c] + pos_cas[i] + (pos_cas[i]-enc_pos[i])*scale

            data_c = cube_instance[0].data
            colors = plt.get_cmap('gray')(norm(data_c[i1, :, :].flatten()))

            for p in range(len(rec_vertices)):
                pixel_array = rec_vertices[p]
                pixel_color = colors[p]
                traces_basic.append(create_pixel_mesh(pixel_array, pixel_color))


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
    if vims_cubes:
        for i in range(len(frames)):
            step = {"args": [
                [f"frame_{i}"],
                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate", "transition": {"duration": 100}}
            ],
                # "label": times_filtered_utc[i],
                "label": f"{times_filtered_utc[i]} - cube: {cube_names[i]}",
                # Display time step and corresponding value
                "method": "animate"}
            slider_steps.append(step)

    else:
        for i in range(len(frames)):
            step = {"args": [
                [f"frame_{i}"],
                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate", "transition": {"duration": 100}}
            ],
                # "label": times_filtered_utc[i],
                "label": f"{times_filtered_utc[i]}",  # Display time step and corresponding value

                "method": "animate"}
            slider_steps.append(step)

    # Update layout and define buttons
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-0.7, y=0, z=0.2))
    if d_sight:
        if cs_radius < np.linalg.norm(pos_dio[0]) * 1.2:
            axis_range = [-np.linalg.norm(pos_cas[0]) * 1.2, np.linalg.norm(pos_cas[0]) * 1.2]
            # axis_range = [-cs_radius*1.2, cs_radius*1.2]

        else:
            axis_range = [-np.linalg.norm(pos_dio[0]) * 1.2, np.linalg.norm(pos_dio[0]) * 1.2]
            axis_range = [-cs_radius * 1.2, cs_radius * 1.2]
    else:
        if np.linalg.norm(pos_cas[0])>np.linalg.norm(pos_dio[0]):
            axis_range = [-np.linalg.norm(pos_cas[0])*1.2, np.linalg.norm(pos_cas[0])*1.2]
        else:
            axis_range = [-np.linalg.norm(pos_dio[0])*1.2, np.linalg.norm(pos_dio[0])*1.2]

    # axis_range = [-dist * 1.2, dist * 1.2]

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
        }],
        # scene=dict(
        # xaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False,zeroline=False,
        #            showticklabels=False, showaxeslabels=False),
        # yaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False,zeroline=False,
        #            showticklabels=False, showaxeslabels=False),
        # zaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False,zeroline=False,
        #            showticklabels=False, showaxeslabels=False),
        # aspectmode='cube'
        # backgroundcolor="black"
        scene = dict(
        xaxis=dict(range=axis_range, gridwidth=0.5, showgrid=True, showbackground=True, zeroline=True,
                   showticklabels=True, showaxeslabels=True),
        yaxis=dict(range=axis_range, gridwidth=0.5, showgrid=True, showbackground=True, zeroline=True,
                   showticklabels=True, showaxeslabels=True),
        zaxis=dict(range=axis_range, gridwidth=0.5, showgrid=True, showbackground=True, zeroline=True,
                   showticklabels=True, showaxeslabels=True),
        aspectmode='cube'
    ),
        scene_camera=camera,
        template='plotly_dark'
    )
    # # Save each frame as an image
    # # Update the camera settings for each frame
    # fig.update_layout(scene_camera=camera)
    #
    # for i, frame in enumerate(frames):
    #     frame_filename = os.path.join(frames_dir, f'frame_{i}.png')
    #     pio.write_image(frame, frame_filename)

    # write html
    fig.write_html("../docs/satvims_tool_showcase.html")

    # Show figure
    fig.show()


def plot_spice_scene_ring_bb( flyby_name, hours_from_closest_approach, orbit_steps, frame_steps, instrument_name,
                              fov_dist, vims_cubes=None , i1=254, vims_cubes2=None):
    """
    plot_spice_scene ring, not animated with time slider, but shows two boundary cubes from cube list input.
    Centered on Saturn

    Parameters:
    -----------
    flyby_name : str
        The name of the flyby event to visualize.

    hours_from_closest_approach : float
        The number of hours from the closest approach to retrieve the time range for the flyby.

    orbit_steps : int
        The number of steps to divide the orbit time range for plotting.

    frame_steps : int
        The number of steps in seconds between frames for time filtering.

    instrument_name : str
        The name of the instrument to retrieve field of view (FOV) information from SPICE kernels.

    vims_cubes : list of VIMSCube, optional
        A list of VIMS cube objects to plot. If provided, a single cube will be plotted for each time step.

    i1 : int, optional
        The index of the VIMS-IR channel  (default is 254).

    d_sight : float, optional
        The distance to sight for celestial sphere rendering. If not provided, projection at Cassini's position.

    Returns:
    --------
    None
        The function displays the plot directly.

    """

    # ------------------------------------------------ SPICE ----------------------------------------------
    # projection tool is based in the Enceladus centered reference frame
    # use casini and sun vectors observed from enceladus
    # in the J2000 frame to minimize movement of the sun -> assume sun position is constant
    print("------------------ SPICE -----------------------")
    METAKR = "../data/cassMetaK.txt"
    SCLKID = -82
    spice.furnsh(METAKR)

    if vims_cubes:
        times = []
        time_utc = []
        cube_names = []
        for vcube in vims_cubes:
            cube_name = vcube.img_id
            cube_names.append(cube_name)
            # get et time
            time_et_i = vcube.et_start + (vcube.et_stop-vcube.et_start)/2
            # append times to lists
            # time_utc.append(time_utc_i)
            time_utc.append(spice.et2utc(time_et_i, "c", 0))
            times.append(time_et_i)

        times_filtered = times
        times_filtered_utc = time_utc
        print(times_filtered_utc)
    else:
        # select time range and fly-by
        utc_range = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach)
        utc_range_end = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach/2)
        utc_start = utc_range[0]
        utc_end = utc_range_end[1]
        print("start UTC = ", utc_start)
        print("end UTC = ", utc_end)
        et_start = spice.str2et(utc_start)
        et_end = spice.str2et(utc_end)
        print("ET start seconds past J2: {} ".format(et_start))
        print("ET end seconds past J2: {} ".format(et_end))

        # time steps
        step = orbit_steps
        times = np.array([x * (et_end - et_start) / step + et_start for x in range(step)])

        # filter times for those available in the spice camera kernel
        frame_steps = frame_steps  # size of the steps in seconds
        times_filtered = get_remaining_ets(flyby_name, frame_steps, start_et=et_start, end_et=et_end)
        times_filtered_utc = spice.et2utc(times_filtered, "C", 0)

    if vims_cubes2:
        times2 = []
        time_utc2 = []
        cube_names2 = []
        for vcube in vims_cubes2:
            cube_name2 = vcube.img_id
            cube_names2.append(cube_name2)
            # get et time
            time_et_i = vcube.et_start + (vcube.et_stop - vcube.et_start) / 2
            # append times to lists
            # time_utc.append(time_utc_i)
            time_utc2.append(spice.et2utc(time_et_i, "c", 0))
            times2.append(time_et_i)

        pos_cas2, _ = spice.spkpos('CASSINI', times2, 'J2000', 'NONE', 'SATURN')

    # position of the sun as seen from Enceladus in the J2000 reference frame(km)
    pos_sun, ltime = spice.spkpos('SUN', times, 'J2000', 'LT+S', 'SATURN')
    # Position of CASSINI as seen from Enceladus in the J2000 frame (km)
    pos_cas, ltime_cas = spice.spkpos('CASSINI', times, 'J2000', 'NONE', 'SATURN')


    sc_cord = pos_cas
    if sc_cord.ndim == 1:
        # Compute the norm directly
        max_sc_distance = np.linalg.norm(sc_cord)
    else:
        sc_norms = np.linalg.norm(sc_cord, axis=1)  # Compute the norm of each row
        max_sc_distance = np.max(sc_norms)          # Find the maximum norm

    cs_radius = fov_dist                   # set max distance to radius of celestial sphere

    # get mean SHT orientation in the J2000 frame from Enceladus spin-axis in IAU Enceladus frame
    mean_time = np.mean(times)
    pform2 = spice.pxform("IAU_ENCELADUS", 'J2000', float(mean_time))  # get transformation matrix from IAU_E to J200
    plume_axis = np.array([0, 0, -1])  # orientation of SHT in IAU_enceladus
    plume_axis_j2000 = list(spice.mxv(pform2, plume_axis))

    # get all cassini locations, instrument coord systems and fovs for filtered times
    instrument_times = np.array(times_filtered)
    # get fov
    room = 4  # the maximum number of 3-dimensional vectors that can be returned in `bounds'.
    [shape, insfrm, bsight, n, bounds] = spice.getfvn(instrument_name, room)
    bounds = np.vstack((bounds, [0, 0, 0]))  # add the origin of the fov (tip of the pyramid)
    fov_list = []                            # create an empty list to store the fov vertices for each et
    cord_sys_list = []                       # create an empyt lists which stores the coordinate system axes
    bsights = []
    for ins_time in instrument_times:
        iss_point, ltime_iss = spice.spkpos('CASSINI', ins_time, 'J2000', 'NONE', 'SATURN')
        pform = spice.pxform(instrument_name, 'J2000', ins_time)      # rotation matrix for specific et

        bsight_i = spice.mxv(pform, bsight)
        bsights.append(bsight_i)

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

    if vims_cubes2:
        fov_list2= []
        for ins_time in times2:
            iss_point, ltime_iss = spice.spkpos('CASSINI', ins_time, 'J2000', 'NONE', 'SATURN')
            pform = spice.pxform(instrument_name, 'J2000', ins_time)  # rotation matrix for specific et

            bsight_i = spice.mxv(pform, bsight)
            bsights.append(bsight_i)

            # convert Instrument frame bounds to J2000 frame
            bounds_j2000 = np.zeros((5, 3))
            for i in range(len(bounds)):
                bounds_j2000[i] = spice.mxv(pform, bounds[i])

            # needs to be size of cs_sphere
            iss_fov_vertices = bounds_j2000 * cs_radius + iss_point
            fov_list2.append(iss_fov_vertices)

    # get average of sun positions
    mean_sun_cord = np.mean(pos_sun, axis=0)
    print("mean sun coordinate", mean_sun_cord)
    pos_max = pos_sun[-1] - pos_sun[0]
    print("maximum difference in lighting geometry [deg]", angle_between_vectors(pos_sun[-1], pos_sun[0])/2)

    # position of Dione
    pos_dio , ltime_dio =  spice.spkpos("DIONE", instrument_times, 'J2000', 'None', 'SATURN')

    state_dio , ltime_dio =  spice.spkezr("DIONE", instrument_times, 'J2000', 'None', 'SATURN')
    pos_dio_sat = [state[:3] for state in state_dio]
    vel_dio = [state[-3:] for state in state_dio]

    # position ot tethys
    state_tet, ltime_tet = spice.spkezr("TETHYS", instrument_times, 'J2000', 'None', 'SATURN')
    pos_tet = [state[:3] for state in state_tet]
    vel_tet = [state[-3:] for state in state_tet]

    # position phoebe
    days_range = 300
    pho_start = times[0] - 24*60*60*days_range
    pho_end = times[-1] + 24*60*60*days_range
    pho_times = np.linspace( pho_start, pho_end, 1000)
    state_pho, _ = spice.spkezr("PHOEBE", pho_times, 'J2000', 'None', 'SATURN')
    pos_pho = [state[:3] for state in state_pho]

    pos_pho2, _ = spice.spkpos("PHOEBE", instrument_times, 'J2000', 'None', 'SATURN')

    # IApetus
    days_range = 40
    pho_start = times[0] - 24 * 60 * 60 * days_range
    pho_end = times[-1] + 24 * 60 * 60 * days_range
    pho_times = np.linspace(pho_start, pho_end, 1000)
    state_iap, _ = spice.spkezr("IAPETUS", pho_times, 'J2000', 'None', 'SATURN')
    pos_iap = [state[:3] for state in state_iap]
    # vel_pho = [state[-3:] for state in state_pho]

    # position of cassini but at filtered time steps
    pos_cas_fil, ltime_cas_fil = spice.spkpos('CASSINI', instrument_times, 'J2000', 'NONE', 'SATURN')

    # state of enceladus a sseen from cassini
    state_enc, ltimev = spice.spkezr("ENCELADUS", instrument_times, "J2000", "LT+S", "SATURN")
    # Extract the last three values from each array
    enc_vel = [state[-3:]/spice.vnorm(state[-3:]) for state in state_enc]
    enc_pos = [state[:3] for state in state_enc]

    # Clean up the kernels
    spice.kclear()

    # ----- create bodies ------
    # Create traces for each body
    traces = []

    # Saturn
    sat_radius = 58232
    xsat, ysat, zsat = sphere(sat_radius)
    trace_sat = go.Surface(z=zsat, x=xsat, y=ysat, colorscale=[[0, 'tan'], [1, 'tan']], showscale=False,
                                 name="Saturn")
    traces.append(trace_sat)

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
    # print("sun right ascension", sun_right_ascension)
    # print("sun declination", sun_declination)
    # Define rotation matrices for latitude (rotate around y) and longitude (rotate around z)

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
        name='Sun Direction'  # Name of the trace
    )
    traces.append(line_trace_sun)

    # get angle of cone cross-section on image plane
    # rotate the cassini positions so that the sun vector coincides with the x-axis
    # this allows for 2d analysis of the cross-sections
    sc_cordt_fil = pos_cas_fil.T
    # make sure rotations happen for the right signs
    if sun_right_ascension > 0:
        rotated_sc = rotate_z(sc_cordt_fil[0], sc_cordt_fil[1], sc_cordt_fil[2], -sun_right_ascension)
    else:
        rotated_sc = rotate_z(sc_cordt_fil[0], sc_cordt_fil[1], sc_cordt_fil[2], sun_right_ascension)

    if sun_declination > 0:
        rotated_sc = rotate_y(rotated_sc[0], rotated_sc[1], rotated_sc[2], sun_declination)
    else:
        rotated_sc = rotate_y(rotated_sc[0], rotated_sc[1], rotated_sc[2], -sun_declination)


    # plot spacecraft trajectory
    sc_cordt = sc_cord.T
    if sc_cord.ndim == 1:
        trace_sc = go.Scatter3d(x=[sc_cordt[0]], y=[sc_cordt[1]], z=[sc_cordt[2]], mode='markers',
                     marker=dict(size=2, color='skyblue'), name='Cassini position')
    else:
        trace_sc = go.Scatter3d(x=sc_cordt[0], y=sc_cordt[1], z=sc_cordt[2], mode='lines', line=dict(color='skyblue', width=2),
                                name="Cassini trajectory")
    traces.append(trace_sc)

    # Phoebe ring
    pho_cordt = np.array(pos_pho).T
    trace_pho = go.Scatter3d(x=pho_cordt[0], y=pho_cordt[1], z=pho_cordt[2], mode='lines',
                            line=dict(color='grey', width=2, dash='dash'),
                            name="Phoebe trajectory")
    traces.append(trace_pho)
    # Iapetus
    iap_cordt = np.array(pos_iap).T
    trace_iap = go.Scatter3d(x=iap_cordt[0], y=iap_cordt[1], z=iap_cordt[2], mode='lines',
                             line=dict(color='pink', width=2, dash='dash'),
                             name="Iapetus trajectory")
    traces.append(trace_iap)

    # E-ring
    circle_points = generate_circle_points(np.array([0, 0, 0]), np.linalg.norm(np.mean(enc_pos, axis=0)),
                                           np.mean(enc_vel, axis=0), origin=np.mean(enc_pos, axis=0))
    filtered_circle_points = circle_points
    # Add the filtered circle points to the figure
    ering_trace = go.Scatter3d(x=filtered_circle_points[:, 0], y=filtered_circle_points[:, 1],
                               z=filtered_circle_points[:, 2],
                               mode='lines', name='E-ring', line=dict(color='lightgrey', dash='dash'))
    traces.append(ering_trace)

    # Dione
    # trace_dio = go.Scatter3d(x=[pos_dio[i][0]], y=[pos_dio[i][1]], z=[pos_dio[i][2]], mode='markers',
    #                          marker=dict(size=4, color='darkseagreen'), name='Dione')
    # traces.append(trace_dio)

    # dione ring
    dione_ring_radius = np.linalg.norm(np.mean(pos_dio_sat, axis=0))
    circle_points = generate_circle_points(np.array([0, 0, 0]), dione_ring_radius, np.mean(vel_dio,axis=0),
                                           origin=np.mean(pos_dio_sat, axis=0))
    dring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='Dione orbit', line=dict(color='darkseagreen', dash='dash', width=2))
    traces.append(dring_trace)

    # Tethys
    # trace_tet = go.Scatter3d(x=[pos_tet[i][0]], y=[pos_tet[i][1]], z=[pos_tet[i][2]], mode='markers',
    #                          marker=dict(size=4, color='indianred'), name='Tethys')
    # traces.append(trace_tet)

    # tethys ring
    tet_ring_radius = np.linalg.norm(np.mean(pos_tet, axis=0))
    circle_points = generate_circle_points(np.array([0, 0, 0]), tet_ring_radius, np.mean(vel_tet,axis=0),
                                           origin=np.mean(pos_tet, axis=0))
    tring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='Tethys orbit', line=dict(color='indianred', dash='dash'))
    traces.append(tring_trace)

    # phoebe
    trace_pho = go.Scatter3d(x=[pos_pho2[i][0]], y=[pos_pho2[i][1]], z=[pos_pho2[i][2]], mode='markers',
                             marker=dict(size=5, color='grey'), name='Phoebe')
    traces.append(trace_pho)


    # Saturn F-ring
    fring_radius = 140220
    circle_points = generate_circle_points(np.array([0, 0, 0]), fring_radius, enc_vel[0], origin=enc_pos[0])
    fring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='F-ring', line=dict(color='moccasin'))
    traces.append(fring_trace)
    # Saturn G-ring
    gring_radius = 349554 / 2
    circle_points = generate_circle_points(np.array([0, 0, 0]), gring_radius, enc_vel[0], origin=enc_pos[0])
    gring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='G-ring', line=dict(color='peru'))
    traces.append(gring_trace)


    if vims_cubes:
        # VIMS colorscale
        data = np.array([
            cube.data
            for cube in vims_cubes
        ])
        norm = Normalize(vmin=0, vmax=np.percentile(data[:, i1, :, :], 99), clip=True)

        time_start_end = [times_filtered[0], times_filtered[-1]]
        for i in range(len(time_start_end)):

            if i == 0:
                k = 0
            else:
                k =-1

            cube_instance = [vims_cubes[k]]

            paths = [
                Path(cube.rsky[:, l, s, :].T)
                for cube in cube_instance
                for l in range(cube.NL)
                for s in range(cube.NS)
            ]

            vertices = np.stack([
                path.vertices
                for path in paths
            ])

            scale = 10
            dist = np.linalg.norm((pos_cas[k] - enc_pos[k]) * scale)
            # dist = cs_radius
            dist_along_sight = bsights[k]*cs_radius
            dist =np.linalg.norm(dist_along_sight)
            # dist = np.linalg.norm(pos_dio[0])
            # dist = np.linalg.norm(enc_pos[i]*scale)

            rec_vertices = []
            for vertice in vertices:
                # a single vertice is an array representing the 4 corner coordinates of a single pixel
                rec_vertice = np.zeros((4, 3))
                for r in range(len(vertice)):
                    # convert to rectilinear coordinates
                    x, y, z = spice.radrec(dist, np.deg2rad(vertice[r, 0]), np.deg2rad(vertice[r, 1]))
                    rec_vertice[r] = np.array([x, y, z])
                # store new rectilinear pixel vertice in list containing all pixels of a single cube
                rec_vertices.append(rec_vertice)

            # Calculate the centroid of the combined image
            all_rec_vertices = np.array(rec_vertices).reshape(-1, 3)
            centroid = all_rec_vertices.mean(axis=0)

            # shift pixels first to center of the plot (Saturn) and then to Cassini's position
            for rec_ver in rec_vertices:
                for c in range(len(rec_ver)):
                    rec_ver[c] = rec_ver[c] + pos_cas[k]

            data_c = cube_instance[0].data
            colors = plt.get_cmap('gray')(norm(data_c[i1, :, :].flatten()))

            for p in range(len(rec_vertices)):
                pixel_array = rec_vertices[p]
                pixel_color = colors[p]
                traces.append(create_pixel_mesh(pixel_array, pixel_color))


    # get list of traces for start and end frame
    # fov traces
    for i in [0, -1]:
        # single frame fov
        v = fov_list[i]
        # Generate list of sides' polygons of the pyramid
        faces = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
                 [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

        # Extract x, y, z coordinates of each vertex
        x, y, z = v[:, 0], v[:, 1], v[:, 2]

        # Cassini position in each frame
        trace_tip = go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers',
                                marker=dict(size=3, color='skyblue'), name='Cassini position')
        traces.append(trace_tip)

        for face in faces:
            x_face = [vertex[0] for vertex in face]
            y_face = [vertex[1] for vertex in face]
            z_face = [vertex[2] for vertex in face]
            trace_face = go.Mesh3d(x=x_face, y=y_face, z=z_face, color='cyan', opacity=0.25, showlegend=False)
            traces.append(trace_face)

        # Enceladus
        trace_enc = go.Scatter3d(x=[enc_pos[i][0]], y=[enc_pos[i][1]], z=[enc_pos[i][2]], mode='markers',
                                 marker=dict(size=4, color='lightgrey'), name='Enceladus')
        traces.append(trace_enc)

    # Create figure
    fig = go.Figure()

    # Add traces
    for trace in traces:
        fig.add_trace(trace)

    # Update layout and define camera settings
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-0.5, y=-1, z=0.1)
    )
    if cs_radius< np.linalg.norm(pos_dio[0]) * 1.2:
        axis_range = [-np.linalg.norm(pos_cas[0]) * 1.2, np.linalg.norm(pos_cas[0]) * 1.2]
        # axis_range = [-cs_radius*1.2, cs_radius*1.2]

    else:
        axis_range = [-np.linalg.norm(pos_dio[0]) * 1.2, np.linalg.norm(pos_dio[0]) * 1.2]
        axis_range = [-cs_radius*1.2, cs_radius*1.2]

    fig.update_layout(
        title=f'Enceladus fly-by {flyby_name}',
        scene=dict(
            xaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
                       showticklabels=False, showaxeslabels=False),
            yaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
                       showticklabels=False, showaxeslabels=False),
            zaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
                       showticklabels=False, showaxeslabels=False),
            aspectmode='cube'
        ),
        scene_camera=camera,
        template='plotly_dark'
    )

    # fig.update_layout(
    #     title=f'Enceladus fly-by {flyby_name}',
    #     scene=dict(
    #         xaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=True,
    #                    showticklabels=True, showaxeslabels=True),
    #         yaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=True,
    #                    showticklabels=True, showaxeslabels=True),
    #         zaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
    #                    showticklabels=True, showaxeslabels=True),
    #         aspectmode='cube'
    #     ),
    #     scene_camera=camera,
    #     template='plotly_dark'
    # )

    # Show figure
    fig.show()


def plot_spice_scene_ring_bb_cam( flyby_name, hours_from_closest_approach, orbit_steps, frame_steps, instrument_name,
                              fov_dist, vims_cubes=None , i1=254):
    """
        plot_spice_scene ring, not animated with time slider, but shows two boundary cubes from cube list input.
        Centered on VIMS observations

        Parameters:
        -----------
        flyby_name : str
            The name of the flyby event to visualize.

        hours_from_closest_approach : float
            The number of hours from the closest approach to retrieve the time range for the flyby.

        orbit_steps : int
            The number of steps to divide the orbit time range for plotting.

        frame_steps : int
            The number of steps in seconds between frames for time filtering.

        instrument_name : str
            The name of the instrument to retrieve field of view (FOV) information from SPICE kernels.

        vims_cubes : list of VIMSCube, optional
            A list of VIMS cube objects to plot. If provided, a single cube will be plotted for each time step.

        i1 : int, optional
            The index of the VIMS-IR channel  (default is 254).

        d_sight : float, optional
            The distance to sight for celestial sphere rendering. If not provided, projection at Cassini's position.

        Returns:
        --------
        None
            The function displays the plot directly.

        """
    # ------------------------------------------------ SPICE ----------------------------------------------
    # projection tool is based in the Enceladus centered reference frame
    # use casini and sun vectors observed from enceladus
    # in the J2000 frame to minimize movement of the sun -> assume sun position is constant
    print("------------------ SPICE -----------------------")
    METAKR = "../data/cassMetaK.txt"
    SCLKID = -82
    spice.furnsh(METAKR)

    # ----- create bodies ------
    # Create traces for each body
    traces = []

    if vims_cubes:
        times = []
        time_utc = []
        cube_names = []
        for vcube in vims_cubes:
            cube_name = vcube.img_id
            cube_names.append(cube_name)
            # get et time
            time_et_i = vcube.et_start + (vcube.et_stop-vcube.et_start)/2
            # append times to lists
            # time_utc.append(time_utc_i)
            time_utc.append(spice.et2utc(time_et_i, "c", 0))
            times.append(time_et_i)

        times_filtered = times
        times_filtered_utc = time_utc
        print(times_filtered_utc)
    else:
        # select time range and fly-by
        utc_range = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach)
        utc_range_end = get_time_flyby(flyby_name, "../data/flybys.txt", hours_from_closest_approach/2)
        utc_start = utc_range[0]
        utc_end = utc_range_end[1]
        print("start UTC = ", utc_start)
        print("end UTC = ", utc_end)
        et_start = spice.str2et(utc_start)
        et_end = spice.str2et(utc_end)
        print("ET start seconds past J2: {} ".format(et_start))
        print("ET end seconds past J2: {} ".format(et_end))

        # time steps
        step = orbit_steps
        times = np.array([x * (et_end - et_start) / step + et_start for x in range(step)])

        # filter times for those available in the spice camera kernel
        frame_steps = frame_steps  # size of the steps in seconds
        times_filtered = get_remaining_ets(flyby_name, frame_steps, start_et=et_start, end_et=et_end)
        times_filtered_utc = spice.et2utc(times_filtered, "C", 0)

    # position of the sun as seen from Enceladus in the J2000 reference frame(km)
    pos_sun, ltime = spice.spkpos('SUN', times, 'J2000', 'LT+S', 'SATURN')
    # Position of CASSINI as seen from Enceladus in the J2000 frame (km)
    pos_cas, ltime_cas = spice.spkpos('CASSINI', times, 'J2000', 'NONE', 'SATURN')


    sc_cord = pos_cas
    if sc_cord.ndim == 1:
        # Compute the norm directly
        max_sc_distance = np.linalg.norm(sc_cord)
    else:
        sc_norms = np.linalg.norm(sc_cord, axis=1)  # Compute the norm of each row
        max_sc_distance = np.max(sc_norms)          # Find the maximum norm

    cs_radius = fov_dist                   # set max distance to radius of celestial sphere

    # get mean SHT orientation in the J2000 frame from Enceladus spin-axis in IAU Enceladus frame
    mean_time = np.mean(times)
    pform2 = spice.pxform("IAU_ENCELADUS", 'J2000', float(mean_time))  # get transformation matrix from IAU_E to J200
    plume_axis = np.array([0, 0, -1])  # orientation of SHT in IAU_enceladus
    plume_axis_j2000 = list(spice.mxv(pform2, plume_axis))

    # get all cassini locations, instrument coord systems and fovs for filtered times
    instrument_times = np.array(times_filtered)
    # get fov
    room = 4  # the maximum number of 3-dimensional vectors that can be returned in `bounds'.
    [shape, insfrm, bsight, n, bounds] = spice.getfvn(instrument_name, room)
    bounds = np.vstack((bounds, [0, 0, 0]))  # add the origin of the fov (tip of the pyramid)
    fov_list = []                            # create an empty list to store the fov vertices for each et
    cord_sys_list = []                       # create an empyt lists which stores the coordinate system axes
    bsights = []
    for ins_time in instrument_times:
        iss_point, ltime_iss = spice.spkpos('CASSINI', ins_time, 'J2000', 'NONE', 'SATURN')
        pform = spice.pxform(instrument_name, 'J2000', ins_time)      # rotation matrix for specific et

        bsight_i = spice.mxv(pform, bsight)
        bsights.append(bsight_i)

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

    # position of Dione
    pos_dio , ltime_dio =  spice.spkpos("DIONE", instrument_times, 'J2000', 'None', 'SATURN')

    state_dio , ltime_dio =  spice.spkezr("DIONE", instrument_times, 'J2000', 'None', 'SATURN')
    pos_dio_sat = [state[:3] for state in state_dio]
    vel_dio = [state[-3:] for state in state_dio]

    # position ot tethys
    state_tet, ltime_tet = spice.spkezr("TETHYS", instrument_times, 'J2000', 'None', 'SATURN')
    pos_tet = [state[:3] for state in state_tet]
    vel_tet = [state[-3:] for state in state_tet]

    # position phoebe
    days_range = 275
    pho_start = times[0] - 24*60*60*days_range
    pho_end = times[-1] + 24*60*60*days_range
    pho_times = np.linspace( pho_start, pho_end, 1000)
    state_pho, _ = spice.spkezr("PHOEBE", pho_times, 'J2000', 'None', 'SATURN')
    pos_pho = [state[:3] for state in state_pho]

    pos_pho2, _ = spice.spkpos("PHOEBE", instrument_times, 'J2000', 'None', 'SATURN')

    # try some irregular moons
    # gridr = "654"
    # # S2004 52 = "65152"
    # S2004 S13 = "65041"
    irr_id = "642"
    days_range = 600
    pho_start = times[0] - 24 * 60 * 60 * days_range
    pho_end = times[-1] + 24 * 60 * 60 * days_range
    pho_times = np.linspace(pho_start, pho_end, 1000)
    spice_ids_retro = [
    609, 643, 639, 627, 65055, 648, 644, 65050, 65056, 623,
    630, 65041, 625, 650, 65045, 65040, 65041, 640, 65035,
    644, 638, 642, 645, 65048, 641, 646, 647, 619, 652
    ]
    spice_ids_retro = [65041]
    for spice_id in spice_ids_retro:
        pos_irr, _ = spice.spkpos(str(spice_id), pho_times, 'J2000', 'None', 'SATURN')
        irr_cordt = np.array(pos_irr).T
        trace_irr = go.Scatter3d(x=irr_cordt[0], y=irr_cordt[1], z=irr_cordt[2], mode='lines',
                                 line=dict(color='grey', width=1),
                                 name=f"Ir:{spice_id}")
        traces.append(trace_irr)

    # pos_irr, _ = spice.spkpos(irr_id, instrument_times, 'J2000', 'None', 'SATURN')

    # exit()
    # IApetus
    days_range = 40
    pho_start = times[0] - 24 * 60 * 60 * days_range
    pho_end = times[-1] + 24 * 60 * 60 * days_range
    pho_times = np.linspace(pho_start, pho_end, 1000)
    state_iap, _ = spice.spkezr("IAPETUS", pho_times, 'J2000', 'None', 'SATURN')
    pos_iap = [state[:3] for state in state_iap]
    # vel_pho = [state[-3:] for state in state_pho]

    # position of cassini but at filtered time steps
    pos_cas_fil, ltime_cas_fil = spice.spkpos('CASSINI', instrument_times, 'J2000', 'NONE', 'SATURN')

    # state of enceladus a sseen from cassini
    state_enc, ltimev = spice.spkezr("ENCELADUS", instrument_times, "J2000", "LT+S", "SATURN")
    # Extract the last three values from each array
    enc_vel = [state[-3:]/spice.vnorm(state[-3:]) for state in state_enc]
    enc_pos = [state[:3] for state in state_enc]

    # Clean up the kernels
    spice.kclear()

    # Saturn
    sat_radius = 58232
    xsat, ysat, zsat = sphere(sat_radius)
    trace_sat = go.Surface(z=zsat, x=xsat, y=ysat, colorscale=[[0, 'tan'], [1, 'tan']], showscale=False,
                                 name="Saturn")
    traces.append(trace_sat)

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
    # print("sun right ascension", sun_right_ascension)
    # print("sun declination", sun_declination)
    # Define rotation matrices for latitude (rotate around y) and longitude (rotate around z)

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
        name='Sun Direction'  # Name of the trace
    )
    traces.append(line_trace_sun)

    # get angle of cone cross-section on image plane
    # rotate the cassini positions so that the sun vector coincides with the x-axis
    # this allows for 2d analysis of the cross-sections
    sc_cordt_fil = pos_cas_fil.T
    # make sure rotations happen for the right signs
    if sun_right_ascension > 0:
        rotated_sc = rotate_z(sc_cordt_fil[0], sc_cordt_fil[1], sc_cordt_fil[2], -sun_right_ascension)
    else:
        rotated_sc = rotate_z(sc_cordt_fil[0], sc_cordt_fil[1], sc_cordt_fil[2], sun_right_ascension)

    if sun_declination > 0:
        rotated_sc = rotate_y(rotated_sc[0], rotated_sc[1], rotated_sc[2], sun_declination)
    else:
        rotated_sc = rotate_y(rotated_sc[0], rotated_sc[1], rotated_sc[2], -sun_declination)


    # plot spacecraft trajectory
    sc_cordt = sc_cord.T
    if sc_cord.ndim == 1:
        trace_sc = go.Scatter3d(x=[sc_cordt[0]], y=[sc_cordt[1]], z=[sc_cordt[2]], mode='markers',
                     marker=dict(size=2, color='skyblue'), name='Cassini position')
    else:
        trace_sc = go.Scatter3d(x=sc_cordt[0], y=sc_cordt[1], z=sc_cordt[2], mode='lines', line=dict(color='skyblue', width=2),
                                name="Cassini trajectory")
    traces.append(trace_sc)

    # Phoebe ring
    pho_cordt = np.array(pos_pho).T
    trace_pho = go.Scatter3d(x=pho_cordt[0], y=pho_cordt[1], z=pho_cordt[2], mode='lines',
                            line=dict(color='grey', width=2, dash='dash'),
                            name="Phoebe trajectory")
    trace_pho = go.Scatter3d(x=pho_cordt[0], y=pho_cordt[1], z=pho_cordt[2], mode='lines',
                             line=dict(color='blue', width=4, dash='dash'),
                             name="Phoebe trajectory")
    traces.append(trace_pho)


    # S2004_S52
    # trace_irr = go.Scatter3d(x=[pos_irr[0][0]], y=[pos_irr[0][1]], z=[pos_irr[0][2]], mode='markers',
    #                          marker=dict(size=5, color='grey'), name='S2004_S52')
    # traces.append(trace_irr)

    # Iapetus
    iap_cordt = np.array(pos_iap).T
    trace_iap = go.Scatter3d(x=iap_cordt[0], y=iap_cordt[1], z=iap_cordt[2], mode='lines',
                             line=dict(color='pink', width=2, dash='dash'),
                             name="Iapetus trajectory")
    traces.append(trace_iap)

    # E-ring
    circle_points = generate_circle_points(np.array([0, 0, 0]), np.linalg.norm(np.mean(enc_pos, axis=0)),
                                           np.mean(enc_vel, axis=0), origin=np.mean(enc_pos, axis=0))
    filtered_circle_points = circle_points
    # Add the filtered circle points to the figure
    ering_trace = go.Scatter3d(x=filtered_circle_points[:, 0], y=filtered_circle_points[:, 1],
                               z=filtered_circle_points[:, 2],
                               mode='lines', name='E-ring', line=dict(color='lightgrey', dash='dash'))
    traces.append(ering_trace)

    # dione ring
    dione_ring_radius = np.linalg.norm(np.mean(pos_dio_sat, axis=0))
    circle_points = generate_circle_points(np.array([0, 0, 0]), dione_ring_radius, np.mean(vel_dio,axis=0),
                                           origin=np.mean(pos_dio_sat, axis=0))
    dring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='Dione orbit', line=dict(color='darkseagreen', dash='dash', width=2))
    traces.append(dring_trace)

    # tethys ring
    tet_ring_radius = np.linalg.norm(np.mean(pos_tet, axis=0))
    circle_points = generate_circle_points(np.array([0, 0, 0]), tet_ring_radius, np.mean(vel_tet,axis=0),
                                           origin=np.mean(pos_tet, axis=0))
    tring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='Tethys orbit', line=dict(color='indianred', dash='dash'))
    traces.append(tring_trace)

    # phoebe
    # trace_pho = go.Scatter3d(x=[pos_pho2[i][0]], y=[pos_pho2[i][1]], z=[pos_pho2[i][2]], mode='markers',
    #                          marker=dict(size=5, color='grey'), name='Phoebe')
    # traces.append(trace_pho)


    # Saturn F-ring
    fring_radius = 140220
    circle_points = generate_circle_points(np.array([0, 0, 0]), fring_radius, enc_vel[0], origin=enc_pos[0])
    fring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='F-ring', line=dict(color='moccasin'))
    traces.append(fring_trace)
    # Saturn G-ring
    gring_radius = 349554 / 2
    circle_points = generate_circle_points(np.array([0, 0, 0]), gring_radius, enc_vel[0], origin=enc_pos[0])
    gring_trace = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1],
                               z=circle_points[:, 2],
                               mode='lines', name='G-ring', line=dict(color='peru'))
    traces.append(gring_trace)


    if vims_cubes:
        # VIMS colorscale
        data = np.array([
            cube.data
            for cube in vims_cubes
        ])
        norm = Normalize(vmin=0, vmax=np.percentile(data[:, i1, :, :], 99), clip=True)

        time_start_end = [times_filtered[0], times_filtered[-1]]
        centroids = []
        for i in range(len(time_start_end)):

            if i == 0:
                k = 0
            else:
                k =-1

            cube_instance = [vims_cubes[k]]

            paths = [
                Path(cube.rsky[:, l, s, :].T)
                for cube in cube_instance
                for l in range(cube.NL)
                for s in range(cube.NS)
            ]

            vertices = np.stack([
                path.vertices
                for path in paths
            ])

            dist_along_sight = bsights[k]*cs_radius
            dist =np.linalg.norm(dist_along_sight)

            rec_vertices = []
            for vertice in vertices:
                # a single vertice is an array representing the 4 corner coordinates of a single pixel
                rec_vertice = np.zeros((4, 3))
                for r in range(len(vertice)):
                    # convert to rectilinear coordinates
                    x, y, z = spice.radrec(dist, np.deg2rad(vertice[r, 0]), np.deg2rad(vertice[r, 1]))
                    rec_vertice[r] = np.array([x, y, z])
                # store new rectilinear pixel vertice in list containing all pixels of a single cube
                rec_vertices.append(rec_vertice)

            # Calculate the centroid of the combined image
            all_rec_vertices = np.array(rec_vertices).reshape(-1, 3)
            centroid = all_rec_vertices.mean(axis=0)
            centroids.append(centroid)

            # shift pixels to start from Cassini's position
            for rec_ver in rec_vertices:
                for c in range(len(rec_ver)):
                    rec_ver[c] = rec_ver[c] + pos_cas[k]

            data_c = cube_instance[0].data
            colors = plt.get_cmap('gray')(norm(data_c[i1, :, :].flatten()))

            for p in range(len(rec_vertices)):
                pixel_array = rec_vertices[p]
                pixel_color = colors[p]
                traces.append(create_pixel_mesh(pixel_array, pixel_color))

            # get lines along distance to see intersection
            distances = np.linspace(0, dist, 10)
            sky_cord = np.array([8.2, -6.1])  # point on brightband
            line_cords = []
            for d in distances:
                x, y, z = spice.radrec(d, np.deg2rad(sky_cord[0]), np.deg2rad(sky_cord[1]))
                line_cord = np.array([x, y, z]) + pos_cas[k]
                line_cords.append(line_cord)

            line_cords = np.array(line_cords).T
            line_trace = go.Scatter3d(x=line_cords[0], y=line_cords[1], z=line_cords[2], mode='lines',
                                      line=dict(color='magenta', width=1),
                                      name="line of sight 1 sky cord")
            # traces.append(line_trace)

        # triangulated E19 bb
        if flyby_name=="E19":
            x_range = np.linspace(8, 10, 10)        # range of ra's to plot triangulated E19 bb
            # line parameters from mosaic
            a = 0.78410015810818
            b = -13.040190472338935
            line_pos_list = []
            for x in x_range:
                y = a*x + b
                x_bb, y_bb, z_bb = spice.radrec(dist, np.deg2rad(x), np.deg2rad(y))
                pos_line_i = np.array([x_bb, y_bb, z_bb])
                pos_line_i = pos_line_i + pos_cas[-1]
                line_pos_list.append(pos_line_i)
            line_array = np.array(line_pos_list).T
            trace_line = go.Scatter3d(x=line_array[0], y=line_array[1], z=line_array[2], mode='lines',
                                     line=dict(color='white', width=2),
                                     name="Triangulated bright band")
            traces.append(trace_line)


    # get list of traces for start and end frame
    # fov traces
    for i in [0, -1]:
        # single frame fov
        v = fov_list[i]
        # Generate list of sides' polygons of the pyramid
        faces = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
                 [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

        # Extract x, y, z coordinates of each vertex
        x, y, z = v[:, 0], v[:, 1], v[:, 2]

        # Cassini position in each frame
        trace_tip = go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers',
                                marker=dict(size=3, color='skyblue'), name='Cassini position')
        traces.append(trace_tip)

        for face in faces:
            x_face = [vertex[0] for vertex in face]
            y_face = [vertex[1] for vertex in face]
            z_face = [vertex[2] for vertex in face]
            trace_face = go.Mesh3d(x=x_face, y=y_face, z=z_face, color='cyan', opacity=0.25, showlegend=False)
            traces.append(trace_face)

        # Enceladus
        trace_enc = go.Scatter3d(x=[enc_pos[i][0]], y=[enc_pos[i][1]], z=[enc_pos[i][2]], mode='markers',
                                 marker=dict(size=4, color='lightgrey'), name='Enceladus')
        traces.append(trace_enc)

        # # Camera
        centroid_avg = (centroids[0] + centroids[-1]) / 2 + np.mean(pos_cas_fil, axis=0)

        trace_cam = go.Scatter3d(x=[centroid_avg[0]], y=[centroid_avg[1]], z=[centroid_avg[2]], mode='markers',
                                 marker=dict(size=4, color='red'), name='camera center')
        # traces.append(trace_cam)

    # Create figure
    fig = go.Figure()

    # Add traces
    for trace in traces:
        fig.add_trace(trace)

    if cs_radius< np.linalg.norm(pos_dio[0]) * 1.2:
        axis_range = [-np.linalg.norm(pos_cas[0]) * 1.2, np.linalg.norm(pos_cas[0]) * 1.2]
        # axis_range = [-cs_radius*1.2, cs_radius*1.2]
    else:
        axis_range = [-np.linalg.norm(pos_dio[0]) * 1.2, np.linalg.norm(pos_dio[0]) * 1.2]
        multiplier= 2
        axis_range = [-cs_radius*multiplier, cs_radius*multiplier]

    # axis_range = [-np.linalg.norm(pos_pho[0], axis=0)*1.25, np.linalg.norm(pos_pho[0], axis=0)*1.25]

    # Update layout and define camera settings
    camera = dict(
        up=dict(x=0, y=0, z=1),  # afblijven z points up
        center=dict(x=0, y=0, z=0),  # center of focus of the camera
        eye=dict(x=-0.5, y=-1, z=0.1)  # position of the camera
    )
    # print(pos_pho[0])
    axis_mag = axis_range[1]*2

    # print("cass mean = ",np.mean(pos_cas_fil, axis=0))
    # print(centroids[0], centroids[-1])
    # print((centroids[0] + centroids[-1])/2)
    centroid_avg = (centroids[0] + centroids[-1]) / 2 + np.mean(pos_cas_fil, axis=0)
    # print("centroid avg=", centroid_avg)
    # print("axis magntiude=", axis_mag)
    centroid_eye = centroid_avg/ (axis_mag)

    bsight_avg =  (bsights[0] + bsights[-1])/2
    before_centroid_eye = centroid_eye - bsight_avg/70
    # print(bsight_avg)
    # print(centroid_eye)

    cas_dist_diff = np.linalg.norm(pos_cas_fil[0]-pos_cas_fil[1])# - np.linalg.norm(pos_cas_fil[1])
    enc_dist_diff = np.linalg.norm(enc_pos[0]-enc_pos[1]) #- np.linalg.norm(enc_pos[1])
    # print("cassini travelled distance=", cas_dist_diff)
    # print("Enceladus travelled distance=", enc_dist_diff)
    # print("cas y-distance travelled", pos_cas_fil[0][1]-pos_cas_fil[1][1] )
    # print("enc y-distance travelled",enc_pos[0][1]-enc_pos[1][1] )
    # print("cas x-distance travelled", pos_cas_fil[0][0] - pos_cas_fil[1][0])
    # print("enc x-distance travelled", enc_pos[0][0] - enc_pos[1][0])
    # exit()
    pos_c = dict(x=before_centroid_eye[0], y=before_centroid_eye[1], z=before_centroid_eye[2])
    center_c = dict(x=centroid_eye[0], y=centroid_eye[1], z=centroid_eye[2])
    avg_pos_cas = np.mean(pos_cas_fil, axis=0)/axis_mag
    # pos_c = dict(x=avg_pos_cas[0],y=avg_pos_cas[1], z=avg_pos_cas[2] )
    # center_c = dict(x=0, y=0, z=0)
    camera = dict(
        up=dict(x=0, y=0, z=1),  # afblijven z points up
        center=center_c,  # center of focus of the camera
        eye=pos_c  # position of the camera
    )

    fig.update_layout(
        title=f'Enceladus fly-by {flyby_name}',
        scene=dict(
            xaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
                       showticklabels=False, showaxeslabels=False),
            yaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
                       showticklabels=False, showaxeslabels=False),
            zaxis=dict(range=axis_range, gridwidth=0.5, showgrid=False, showbackground=False, zeroline=False,
                       showticklabels=False, showaxeslabels=False),
            aspectmode='cube'
        ),
        scene_camera=camera,
        template='plotly_dark'
    )

    # fig.update_layout(
    #     title=f'Enceladus fly-by {flyby_name}',
    #     scene=dict(
    #         xaxis=dict(range=axis_range, gridwidth=0.5, showgrid=True, showbackground=False, zeroline=True,
    #                    showticklabels=True, showaxeslabels=True),
    #         yaxis=dict(range=axis_range, gridwidth=0.5, showgrid=True, showbackground=False, zeroline=True,
    #                    showticklabels=True, showaxeslabels=True),
    #         zaxis=dict(range=axis_range, gridwidth=0.5, showgrid=True, showbackground=False, zeroline=False,
    #                    showticklabels=True, showaxeslabels=True),
    #         aspectmode='cube'
    #     ),
    #     scene_camera=camera,
    #     template='plotly_dark'
    # )

    # Show figure
    fig.show()


if __name__ == "__main__":

    # load E17 cubes from Nantes VIMS Data Portal
    CUBE_IDS = [
        '1711536135_1', '1711536423_1', '1711536777_1', '1711537065_1', '1711537413_1',
        '1711537701_1', '1711538046_1', '1711538334_1', '1711538684_1', '1711538972_1',
        '1711539317_1', '1711539605_1', '1711539953_1', '1711540241_1', '1711540588_1',
        '1711540876_1', '1711541224_1', '1711541512_1', '1711541857_1', '1711542145_1',
        '1711542490_1', '1711542778_1', '1711543130_1', '1711543418_1', '1711543762_1',
        '1711544050_1', '1711544399_1', '1711544687_1', '1711545033_1', '1711545321_1',
        '1711545666_1', '1711545954_1', '1711546301_1', '1711546589_1', '1711546939_1',
        '1711547227_1', '1711547571_1', '1711547859_1', '1711548206_1', '1711548494_1',
        '1711548846_1', '1711549134_1', '1711549477_1', '1711549765_1', '1711550111_1',
        '1711550399_1', '1711550750_1', '1711551038_1', '1711551375_1', '1711551663_1',
        '1711552021_1', '1711552309_1', '1711552654_1', '1711552942_1', '1711553290_1',
        '1711553578_1',
        # "1711553950_1",
        # "1711554022_1"
    ]

    CUBES = [VIMS(cube_id, root="../data/cubes/") for cube_id in CUBE_IDS]

    # E17 bright band observations
    # CUBE_IDS_bb = ['1711545666_1', '1711545954_1', '1711546301_1', '1711546589_1', '1711546939_1',
    #                 '1711547227_1', '1711547571_1']
    # CUBES_bb = [VIMS(cube_id, root="../data/cubes/") for cube_id in CUBE_IDS_bb]

    # ---------------- example plots ------------------

    # select VIMS wavelength channel
    i1 = 0    # 0.89 μm
    # i1 = 134  # 3.11 μm
    # i1 = 254  # 5.11 μm

    # selected projection distance of VIMS cubes
    rs = 3
    sight = 60330 * rs              # distance in saturn raddii
    sight = None                    # project at cassini location

    # plot saturn centered view with interactive slider between observations (based on VIMS)
    plot_spice_scene_ring(flyby_name="E17", hours_from_closest_approach=8, orbit_steps=4000, frame_steps=600,
                          instrument_name='CASSINI_VIMS_IR',vims_cubes=CUBES, i1=i1, d_sight=sight)

    # same but based on input time range
    # plot_spice_scene_ring(flyby_name="E17", hours_from_closest_approach=12, orbit_steps=4000, frame_steps=600,
    #                       instrument_name='CASSINI_VIMS_IR', vims_cubes=None, i1=i1, d_sight=sight)

    # plot single plot with two cube observations at the same time (to check stripe alignment) - camera center: Saturn
    # plot_spice_scene_ring_bb(flyby_name="E17", hours_from_closest_approach=12, orbit_steps=4000, frame_steps=600,
    #                       instrument_name='CASSINI_VIMS_IR',vims_cubes=CUBES_bb, i1=10, fov_dist=sight
    #                           )

    # plot single plot with two cube observations at the same time (to check stripe alignment) - camera center: VIMS
    # plot_spice_scene_ring_bb_cam(flyby_name="E17", hours_from_closest_approach=12, orbit_steps=4000, frame_steps=600,
    #                              instrument_name='CASSINI_VIMS_IR', vims_cubes=CUBES_bb, i1=10, fov_dist=sight
    #                              )

    # other flyby examples
    # plot_spice_scene_ring(flyby_name="E13", hours_from_closest_approach=12, orbit_steps=4000, frame_steps=600,
    #                       instrument_name='CASSINI_VIMS_IR',vims_cubes=CUBES_e13, i1=134, fov_dist=sight
    #                           )
    # plot_spice_scene_ring(flyby_name="E19", hours_from_closest_approach=12, orbit_steps=4000, frame_steps=600,
    #                       instrument_name='CASSINI_VIMS_IR',vims_cubes=CUBES_e19, i1=144, fov_dist=sight
    #                           )


