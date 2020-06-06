"""
    simple example showing how to dump a number of figures to a single pdf file
"""

from __future__ import print_function

import click
import os.path as op
import sys
import glob
import numpy as np
from scipy import interpolate
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import imageio
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import scipy.ndimage as ndimage
import time

sys.path.append("/Users/samsuidman/Desktop/git_arne/common")
import helpers
import transform


from ipdb import set_trace as db


def get_recordings_mouse(mouse_id):
    # get database paths for session and recordings (interaction/alone conditions)

    recordings_mice = {
        'M3728': {'session': 'M3728/2019_07_09/social_interaction',
                  'interaction': '2019-07-09_15-06-35_Silence_box_no_enclosure_M3729',
                  'alone': '2019-07-09_14-55-57_Silence_box_no_enclosure',
                  'eyelid_left': 'rpi_camera_3DLC_resnet50_M3728_eyelidMar18shuffle1_1030000.h5',
                  'eyelid_right': 'rpi_camera_4DLC_resnet50_M3728_eyelidMar25shuffle1_1030000.h5',
                  'camera_5_mice': 'rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5',
                  'camera_5_box': 'rpi_camera_5DLC_resnet50_M3728_boxFeb14shuffle1_500000.h5',
                  'camera_6_mice': 'rpi_camera_6DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5',
                  'camera_6_box': 'rpi_camera_6DLC_resnet50_M3728_boxFeb14shuffle1_500000.h5'},
        'M3729': {'session': 'M3729/2019_07_08/social_interaction',
                  'interaction': '2019-07-08_12-15-38_Silence_box_no_enclosure_M3728',
                  'alone': '2019-07-08_12-05-16_Silence_box_no_enclosure',
                  'eyelid_left': 'rpi_camera_3DLC_resnet50_M3729_eyelidMar18shuffle1_500000.h5',
                  'eyelid_right': 'rpi_camera_4DLC_resnet50_M3729_eyelidMar18shuffle1_500000.h5',
                  'camera_5_mice': 'rpi_camera_5DLC_resnet50_M3729_miceFeb20shuffle1_1030000.h5',
                  'camera_5_box': 'rpi_camera_5DLC_resnet50_M3729_boxFeb20shuffle1_1030000.h5',
                  'camera_6_mice': 'rpi_camera_6DLC_resnet50_M3729_miceFeb20shuffle1_1030000.h5',
                  'camera_6_box': 'rpi_camera_6DLC_resnet50_M3729_boxFeb20shuffle1_1030000.h5'},
        'M4081': {'session': 'M4081/2019_08_07/social_interaction',
                  'interaction': '2019-08-07_15-59-38_Silence_box_no_enclosure_M3729',
                  'alone': '2019-08-07_15-45-22_Silence_box_no_enclosure',
                  'eyelid_left': 'rpi_camera_3DLC_resnet50_M4081_eyelidMar19shuffle1_1030000.h5',
                  'eyelid_right': 'rpi_camera_4DLC_resnet50_M4081_eyelidMar19shuffle1_1030000.h5',
                  'camera_5_mice': 'rpi_camera_5DLC_resnet50_M4081_miceFeb20shuffle1_1030000.h5',
                  'camera_5_box': 'rpi_camera_5DLC_resnet50_M4081_boxFeb20shuffle1_1030000.h5',
                  'camera_6_mice': 'rpi_camera_6DLC_resnet50_M4081_miceFeb20shuffle1_1030000.h5',
                  'camera_6_box': 'rpi_camera_6DLC_resnet50_M4081_boxFeb20shuffle1_1030000.h5'}
    }

    return recordings_mice[mouse_id]


# use click group to allow for multiple commands (for command line interpreter)


# ------------------------------------------------------------------
# analysis 1: eye closure-aligned traces
# ------------------------------------------------------------------

def load_data_align(db_path, mouse_id,
                    conditions=['interaction', 'alone'],
                    min_likelihood=.99,
                    t_pre=1.,  # time before eye lid closed
                    t_post=1.,  # time after eye lid closed
                    ):

    recordings_mouse = get_recordings_mouse(mouse_id)

    # create time grid for interpolation of traces (for easier averaging afterwards)
    dt = 0.05  # for now, used fixed temporal resolution of 50 ms for interpolation
    n_pre = int(round(t_pre / dt + .5))
    n_post = int(round(t_post / dt + .5))
    t_interp = np.arange(-n_pre, n_post+1) * dt

    data_conditions = {}
    for condition in conditions:

        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse[condition])

        # eye lid data for left/right eyes
        data_cond_eyes = {}
        for i, eye in enumerate(['left', 'right']):

            # read tracked body part data from h5 file
            h5_file = op.join(rec_path, recordings_mouse['eyelid_%s' % eye])
            df = pd.read_hdf(h5_file)

            # get camera time stamps
            camera_data = np.load(h5_file[:h5_file.index('DLC_resnet50_')] + '.npz',
                                  allow_pickle=True)
            cam_ts = camera_data['timestamps']
            n_frames = len(cam_ts)  # there might be more timestamps than actual frames

            # extract data from pandas data frame
            part_names = sorted(set([c[1] for c in df.columns]))
            tracked_parts = {}

            for part_name in part_names:
                dd = {}
                for k in ['x', 'y', 'likelihood']:
                    col = [c for c in df.columns if c[1] == part_name and c[2] == k][0]
                    dd[k] = df[col].to_numpy()[:n_frames]

                # set "bad" values to NaN
                dd['x'][dd['likelihood'] < min_likelihood] = np.NaN
                dd['y'][dd['likelihood'] < min_likelihood] = np.NaN
                tracked_parts[part_name] = dd

            # get eye blink events ("eye_closed")
            ec = tracked_parts['eye_closed']
            indices_closed = np.where(ec['likelihood'] >= min_likelihood)[0]

            # get "first" index for each eye closing sequence (from now on: "event")
            first_ind = indices_closed[np.unique(np.append(0, np.where(np.diff(indices_closed) > 1)[0]+1))]

            # only include eye closing events for which "t_pre" and "t_post" are within the valid range
            # (plus 1 second extra margin to be on the safe side)
            close_ts = cam_ts[first_ind]
            valid = np.logical_and(close_ts >= t_pre + 1,
                                   close_ts <= close_ts[-1] - t_post - 1)
            close_ts = close_ts[valid]

            # get traces around eye closure events
            n_events = len(first_ind)
            traces_parts = {}
            for p in ['lower_edge', 'upper_edge', 'eye_closed']:

                # use linear interpolation to extract trace segments
                f = interpolate.interp1d(cam_ts, np.vstack((tracked_parts[p]['x'],
                                                            tracked_parts[p]['y'])),
                                         axis=1)
                xx = np.zeros((n_events, 2, len(t_interp)))  # (num events, x/y dim, time samples)
                for j, t0 in enumerate(close_ts):
                    xx[j, :, :] = f(t0 + t_interp)

                traces_parts[p] = xx

            data_cond_eyes[eye] = traces_parts

        data_conditions[condition] = data_cond_eyes

    return data_conditions, t_interp


def cli_align(db_path,
              mouse=None):

    mouse_ids = list(mouse)

    colors_dims = {'x': 'tab:blue',
                   'y': 'tab:orange'}

    for mouse_id in mouse_ids:

        data_conditions, t_interp = load_data_align(db_path, mouse_id)

        for j, cond in enumerate(data_conditions):

            fig, axes = plt.subplots(nrows=3,
                                     ncols=2,
                                     sharex=True)
            fig.suptitle('%s (%s)' % (mouse_id, cond))

            data_cond = data_conditions[cond]
            for i, eye in enumerate(['left', 'right']):

                data_eye = data_cond[eye]
                for ii, p in enumerate(sorted(data_eye.keys())):

                    ax = axes[ii, i]
                    xx = data_eye[p]
                    for jj, dim in enumerate(['x', 'y']):
                        ax.plot(t_interp, xx[:, jj, :].T, '-',
                                color=colors_dims[dim])

                    if ii == 2:
                        ax.set_xlabel('Time relative to eye closure (s)')
                    ax.set_ylabel('Position (pix)')

                    ax.set_xticks([-t_interp[0], 0, t_interp[-1]])
                    helpers.set_font_axes(ax)
                    helpers.simple_xy_axes(ax)
                    helpers.adjust_axes(ax)

            fig.tight_layout(rect=(0, 0, 1, .9))

    plt.show(block=True)



# ------------------------------------------------------------------
# analysis 2: y-values of lower/upper eye lids together with eye
# closure events
# ------------------------------------------------------------------

def load_data_traces(db_path, mouse_id,
                     conditions=['interaction', 'alone'],
                     min_likelihood=.99):

    recordings_mouse = get_recordings_mouse(mouse_id)

    data_conditions = {}
    for condition in conditions:

        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse[condition])

        # eye lid data for left/right eyes
        data_cond_eyes = {}
        for i, eye in enumerate(['left', 'right']):

            # read tracked body part data from h5 file
            h5_file = op.join(rec_path, recordings_mouse['eyelid_%s' % eye])
            df = pd.read_hdf(h5_file)

            # get camera time stamps
            camera_data = np.load(h5_file[:h5_file.index('DLC_resnet50_')] + '.npz',
                                  allow_pickle=True)
            cam_ts = camera_data['timestamps']
            n_frames = len(cam_ts)  # there might be more timestamps than actual frames

            # extract data from pandas data frame
            part_names = sorted(set([c[1] for c in df.columns]))
            tracked_parts = {}

            for part_name in part_names:
                dd = {}
                for k in ['x', 'y', 'likelihood']:
                    col = [c for c in df.columns if c[1] == part_name and c[2] == k][0]
                    dd[k] = df[col].to_numpy()[:n_frames]

                # set "bad" values to NaN
                dd['x'][dd['likelihood'] < min_likelihood] = np.NaN
                dd['y'][dd['likelihood'] < min_likelihood] = np.NaN
                tracked_parts[part_name] = dd

            data_cond_eyes[eye] = {'tracked_parts': tracked_parts,
                                   'timestamps': cam_ts}

        data_conditions[condition] = data_cond_eyes

    return data_conditions


def cli_traces(db_path,
               mouse=None):

    mouse_ids = list(mouse)

    colors_edges = {'lower_edge': 'tab:blue',
                    'upper_edge': 'tab:red'}
    min_likelihood = .99

    for mouse_id in mouse_ids:

        data_conditions = load_data_traces(db_path, mouse_id)

        for j, cond in enumerate(data_conditions):

            fig, axes = plt.subplots(nrows=2,
                                     ncols=1,
                                     sharex=True)
            fig.suptitle('%s (%s)' % (mouse_id, cond))

            data_cond = data_conditions[cond]
            for i, eye in enumerate(['left', 'right']):

                tracked_parts = data_cond[eye]['tracked_parts']
                ts = data_cond[eye]['timestamps']

                ax = axes[i]
                ax.set_title('%s eye' % eye.title(),
                             fontweight='bold')

                # plot y-positions of lower and upper edges
                for ii, p in enumerate(['lower_edge', 'upper_edge']):

                    dd = tracked_parts[p]
                    ax.plot(ts, dd['y'], '-',
                            color=colors_edges[p],
                            label=p.replace('_', ' '))

                # indicate periods during which "eye_closed" has been detected
                ec = tracked_parts['eye_closed']
                indices_closed = np.where(ec['likelihood'] >= min_likelihood)[0]

                # get "first" index for each eye closing sequence (from now on: "event")
                first_ind = indices_closed[np.unique(np.append(0, np.where(np.diff(indices_closed) > 1)[0] + 1))]
                for ii, i0 in enumerate(first_ind):
                    i1 = i0
                    while i1+1 < len(ts) and i1+1 in indices_closed:
                        i1 += 1

                    ax.axvspan(ts[i0], ts[i1],
                               facecolor=3*[.5],
                               edgecolor='none',
                               alpha=0.5,  # make semi-transparent
                               zorder=-1  # put behind traces
                               )

                # invert y-axis to account for image y-coordinates increasing towards the bottom
                # (such that "upper" and "lower" eye lids appear at the top and bottom, respectively)
                ax.invert_yaxis()

                # set label, axes limits, fonts etc
                if i == 1:
                    ax.set_xlabel('Time (s)')
                ax.set_ylabel('y-position (pix)')

                ax.set_xlim(ts[0], ts[-1])
                ax.set_xticks([round(ts[0], -1), round(ts[-1], -1)])

                if i == 0:
                    ax.legend(loc='best',
                              fontsize=6).get_frame().set_visible(0)

                helpers.set_font_axes(ax)
                helpers.simple_xy_axes(ax)
                helpers.adjust_axes(ax)

            fig.tight_layout(rect=(0, 0, 1, .9))

    plt.show(block=True)




# ------------------------------------------------------------------
# analysis 3: align "egocentric" markers to eye closure events
# ------------------------------------------------------------------

def load_eyelid_data(rec_path,
                     min_likelihood=.99):
    # almost the same function as "load_data_traces"
    # TODO: revise above code to just call this function

    # get eyelid tracking file information for this mouse
    mouse_id = rec_path.split(op.sep)[-4]
    recordings_mouse = get_recordings_mouse(mouse_id)

    # eye lid data for left/right eyes
    data = {}
    for i, eye in enumerate(['left', 'right']):

        # read tracked body part data from h5 file
        h5_file = op.join(rec_path, recordings_mouse['eyelid_%s' % eye])
        df = pd.read_hdf(h5_file)

        # get camera time stamps
        camera_data = np.load(h5_file[:h5_file.index('DLC_resnet50_')] + '.npz',
                              allow_pickle=True)
        cam_ts = camera_data['timestamps']
        n_frames = len(cam_ts)  # there might be more timestamps than actual frames

        # extract data from pandas data frame
        part_names = sorted(set([c[1] for c in df.columns]))
        tracked_parts = {}

        for part_name in part_names:
            dd = {}
            for k in ['x', 'y', 'likelihood']:
                col = [c for c in df.columns if c[1] == part_name and c[2] == k][0]
                dd[k] = df[col].to_numpy()[:n_frames]

            # set "bad" values to NaN
            dd['x'][dd['likelihood'] < min_likelihood] = np.NaN
            dd['y'][dd['likelihood'] < min_likelihood] = np.NaN
            tracked_parts[part_name] = dd

        data[eye] = {'tracked_parts': tracked_parts,
                     'timestamps': cam_ts}

    return data


def load_eye_closure_data(rec_path,
                          min_likelihood=.99,
                          **kwargs):

    data = load_eyelid_data(rec_path,
                            min_likelihood=min_likelihood,
                            **kwargs)

    # compute vertical distance between upper and lower eye lids
    for i, eye in enumerate(['left', 'right']):

        # note: image y-coordinates are inverted (increase towards the bottom)
        # so need to use negative difference to get easily interpretable differences
        tp = data[eye]['tracked_parts']
        dy = -(tp['upper_edge']['y'] - tp['lower_edge']['y'])
        data[eye]['vertical_eyelid_distance'] = dy

    # get eye closure epochs
    for i, eye in enumerate(['left', 'right']):

        tracked_parts = data[eye]['tracked_parts']
        ts = data[eye]['timestamps']

        # indicate periods during which "eye_closed" has been detected
        ec = tracked_parts['eye_closed']
        indices_closed = np.where(ec['likelihood'] >= min_likelihood)[0]

        # get "first" index for each eye closing sequence (from now on: "event")
        first_ind = indices_closed[np.unique(np.append(0, np.where(np.diff(indices_closed) > 1)[0] + 1))]
        interval_closed = []
        for ii, i0 in enumerate(first_ind):
            i1 = i0
            while i1 + 1 < len(ts) and i1 + 1 in indices_closed:
                i1 += 1

            interval_closed.append((i0, i1))

        data[eye]['eye_closed_interval'] = np.atleast_2d(interval_closed)

    return data


def get_body_part_positions_eye_closed(positions, position_ts, eye_data,
                                       part_names=None):

    if part_names is None:
        part_names = sorted(positions)

    data = {}
    for i, eye in enumerate(['left', 'right']):

        print("  ", eye)

        eye_ts = eye_data[eye]['timestamps']
        intervals = eye_ts[eye_data[eye]['eye_closed_interval']]  # intervals now in seconds instead of video frames

        pos_parts = {}
        for k in part_names:

            print("    ", k)

            # get time points within eye closing intervals
            valid = np.zeros_like(position_ts,
                                  dtype=np.bool)
            for j, (t1, t2) in enumerate(intervals):

                v = np.logical_and(position_ts >= t1,
                                   position_ts <= t2)
                valid[v] = True

            pos_part = positions[k][valid, :]
            pos_parts[k] = pos_part[~np.isnan(pos_part[:, 0]), :]
        data[eye] = pos_parts

    return data

def cli_align_egocentric(db_path,
                         mouse=None):
    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path,
                                                   video='rpi_camera_6',
                                                   min_likelihood=.99,
                                                   unit='cm')

        transformed_positions, xy_centers, angles = transform.transform_egocentric(tracking_data)

        # load eye closure data
        eye_data = load_eye_closure_data(rec_path)

        # extract eye closure-aligned body part positions
        part_names_m2 = sorted([k for k in transformed_positions
                                if k.startswith('m2')])

        pos_eye_closed = get_body_part_positions_eye_closed(transformed_positions,
                                                            tracking_data['timestamps'],
                                                            eye_data,
                                                            part_names=part_names_m2)

        # for each body part, compute a 2D histogram of position during eye closure;
        # for now, use a pre-defined grid width/height (in cm)
        grid_step_size = 2
        width_grid = 20
        height_grid = 20

        x_bins = int(round(width_grid/float(grid_step_size)+.5))
        y_bins = int(round(height_grid / float(grid_step_size) + .5))
        dx = width_grid/2.
        dy = height_grid/2.

        fig, axes = plt.subplots(nrows=2,  # left/right eye
                                 ncols=len(part_names_m2),
                                 sharex=True,
                                 sharey=True)

        for i, eye in enumerate(['left', 'right']):

            for j, part_name in enumerate(part_names_m2):

                xy_pos = pos_eye_closed[eye][part_name]

                ax = axes[i, j]
                if i == 0:
                    ax.set_title('%s' % part_name.replace('_', ' '))

                ax.hist2d(xy_pos[:, 0], xy_pos[:, 1],
                          range=[(-dx, dx),
                                 (-dy, dy)],
                          bins=[x_bins, y_bins])

                # plot cross indicating position of mouse 1 (midpoint between eye cameras)
                ax.plot(0, 0, '+', color='r', ms=10)

                # set axes labels etc
                if i == 1:
                    ax.set_xlabel('x (cm)',
                                  labelpad=2)
                if j == 0:
                    ax.set_ylabel('y (cm)',
                                  labelpad=-2)

                ax.set_xlim(-dx, dx)
                ax.set_ylim(-dy, dy)
                ax.set_xticks([-dx, 0, dx])
                ax.set_yticks([-dy, 0, dy])

                helpers.simple_xy_axes(ax)
                helpers.set_font_axes(ax)
                helpers.adjust_axes(ax, pad=2)

        fig.set_size_inches(8, 3)
        fig.tight_layout(pad=.25,
                         w_pad=.5,
                         h_pad=.5,
                         rect=(.1, 0, 1, 1))

        # add common label for left/right eye to each row of axes
        for i, eye in enumerate(['left', 'right']):
            bbox = axes[i, 0].get_position()
            fig.text(.35*bbox.x0, bbox.y0+.5*bbox.height, eye.title(),
                     ha='center',
                     va='center',
                     family='Arial',
                     fontweight='bold',
                     fontsize=10)

    plt.show(block=True)




# ------------------------------------------------------------------
# analysis 4: plot the distance from each m2 body part to each eye of m1
# ------------------------------------------------------------------



def func_speed_tracking_data(tracking_data):
    body_parts = tracking_data['body_parts'].keys() #make a list for the body parts
    for body_part in body_parts: #look at one bodypart
        position = tracking_data['body_parts'][body_part]['position'] #get the position data for this body_part
        position_difference = np.diff(position,axis=0) #calculate the difference vector of position. This means that this array has one less element.
        velocity_not_same_size = np.transpose([np.transpose(position_difference)[c] / np.diff(tracking_data['timestamps']) for c in range(len(np.transpose(position_difference)))]) #calculate the velocity vector by dividing through the time between two positions
        velocity = np.transpose([np.insert(np.transpose(velocity_not_same_size)[c],0,np.nan) for c in range(len(np.transpose(velocity_not_same_size)))]) #add np.nan in front, so that the first element of the velocity vector is empty. And the second element is the velocity relative to the previous position
        speed = np.asarray([np.sqrt(c[0]**2 +c[1]**2) for c in velocity]) #calculate the speed using the velocity
        speed_averaged = [np.sum(speed[c-5:c+5])/(5*2+1) for c in range(len(speed))] #averaged over 5 frames before and after
        tracking_data['body_parts'][body_part]['speed'] = speed #add the speed to the dictionary of tracking data
        tracking_data['body_parts'][body_part]['speed_averaged'] = speed_averaged
    return tracking_data



def func_speed_eye_data(tracking_data,eye_data):
    body_parts = list(tracking_data['body_parts'].keys())
    for eye in eye_data.keys():
        eye_data[eye]['speed_averaged'] = {}
        eye_data[eye]['closed_eye_speed'] = {}
        for body_part in body_parts:
            speed_averaged_interpolate_f = interpolate.interp1d(tracking_data['timestamps'],tracking_data['body_parts'][body_part]['speed_averaged'],fill_value='extrapolate')
            speed_averaged = speed_averaged_interpolate_f(eye_data[eye]['timestamps'])
            closed_eye_speed = np.asarray([speed_averaged[c[0]:c[1]+1] for c in eye_data[eye]['eye_closed_interval']])
            eye_data[eye]['speed_averaged'][body_part]=speed_averaged
            eye_data[eye]['closed_eye_speed'][body_part] = closed_eye_speed
    return eye_data



def func_abs_distance(tracking_data,eye_data):
    part_names_m2 = sorted([k for k in tracking_data['body_parts'].keys() if k.startswith('m2')])
    for eye in eye_data.keys():
        eye_data[eye]['distance_to_bodypart'] = {}
        eye_data[eye]['closed_eye_distance'] = {}
        for body_part_m2 in part_names_m2:
            distance = np.sqrt(np.sum((tracking_data['body_parts']['m1_eyecam_'+eye]['position'] - tracking_data['body_parts'][body_part_m2]['position']) ** 2, axis=1))
            distance_interpolate_f = interpolate.interp1d(tracking_data['timestamps'],distance,fill_value='extrapolate')
            distance_eye_data = distance_interpolate_f(eye_data[eye]['timestamps'])
            closed_eye_distance = np.asarray([distance_eye_data[c[0]:c[1] + 1] for c in eye_data[eye]['eye_closed_interval']])
            tracking_data['body_parts'][body_part_m2]['distance_to_'+eye+'_eye'] = distance
            eye_data[eye]['distance_to_bodypart'][body_part_m2] = distance_eye_data
            eye_data[eye]['closed_eye_distance'][body_part_m2] = closed_eye_distance
    return tracking_data,eye_data



#this function makes plots of the distance from each m2 body part to each eyecam of m1
def cli_distance_speed_plot(db_path, mouse=None):
    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path,
                                                   video='rpi_camera_6',
                                                   min_likelihood=.99,
                                                   unit='cm')

        #add speed and averaged_speed to the tracking data for each bodypart
        tracking_data = func_speed_tracking_data(tracking_data)

        # load eye closure data
        eye_data = load_eye_closure_data(rec_path)

        #add averaged speed to eye data for each body part and each eye_timestamps
        eye_data = func_speed_eye_data(tracking_data,eye_data)

        #add the distance to m2 body parts to the tracking and eye data. And for the eye data also the closed eye speed.
        tracking_data,eye_data = func_abs_distance(tracking_data,eye_data)

        part_names_m2 = sorted([k for k in tracking_data['body_parts'].keys() if k.startswith('m2')])
        eyes = list(eye_data.keys())

        fig, ax = plt.subplots(nrows=2, ncols=len(part_names_m2), sharex=True, sharey=False, figsize=[20, 5])
        for i in range(len(eyes)):
            for j in range(len(part_names_m2)):
                ax[i][j].scatter(np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][part_names_m2[j]]),
                                 np.concatenate(eye_data[eyes[i]]['closed_eye_speed'][part_names_m2[j]]))
                ax[i][j].set_title('%s' % part_names_m2[j].replace('_', ' '))
                ax[i][j].set_xlabel('y (cm)')
                ax[i][j].set_ylabel('y (cm/s)')
        fig.tight_layout()

    return tracking_data, eye_data






def cli_distance_histogram(db_path,mouse=None):
    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path,
                                                   video='rpi_camera_6',
                                                   min_likelihood=.99,
                                                   unit='cm')

        #add speed and averaged_speed to the tracking data for each bodypart
        tracking_data = func_speed_tracking_data(tracking_data)

        # load eye closure data
        eye_data = load_eye_closure_data(rec_path)

        #add averaged speed to eye data for each body part and each eye_timestamps
        eye_data = func_speed_eye_data(tracking_data,eye_data)

        #add the distance to m2 body parts to the tracking and eye data. And for the eye data also the closed eye speed.
        tracking_data,eye_data = func_abs_distance(tracking_data,eye_data)

        part_names_m2 = sorted([k for k in tracking_data['body_parts'].keys() if k.startswith('m2')])
        eyes = list(eye_data.keys())

        fig, ax = plt.subplots(nrows=2, ncols=len(part_names_m2), sharex=False, sharey=False, figsize=[15, 5])
        for i in range(len(eyes)):
            for j in range(len(part_names_m2)):
                ax[i][j].hist(np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][part_names_m2[j]]), bins=12,
                              range=(0, 30))
                ax[i][j].set_title('%s' % part_names_m2[j].replace('_', ' '))
                ax[i][j].set_xlabel('distance (cm)')
                ax[i][j].set_ylabel('#counts')
                ax[i][j].set_xticks([5, 15, 25])
        fig.tight_layout()
    plt.show(block=True)







def cli_minimal_distance_bodyparts(db_path, mouse=None):
    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path,
                                                   video='rpi_camera_6',
                                                   min_likelihood=.99,
                                                   unit='cm')

        # add speed and averaged_speed to the tracking data for each bodypart
        tracking_data = func_speed_tracking_data(tracking_data)

        # load eye closure data
        eye_data = load_eye_closure_data(rec_path)

        # add averaged speed to eye data for each body part and each eye_timestamps
        eye_data = func_speed_eye_data(tracking_data, eye_data)

        # add the distance to m2 body parts to the tracking and eye data. And for the eye data also the closed eye speed.
        tracking_data, eye_data = func_abs_distance(tracking_data, eye_data)

        eyes = list(eye_data.keys())
        m2_bodyparts = list(eye_data['left']['closed_eye_distance'].keys())

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=[15, 5])
        for i in range(len(eyes)):
            for j in range(2):
                if j == 0:  # minimal distance
                    distance_bodypart_matrix = np.array(
                        [np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][m2_bodypart]) for m2_bodypart in
                         m2_bodyparts])
                    min_distance = np.nanmin(distance_bodypart_matrix, axis=0)
                    ax[i][j].hist(min_distance, bins=16, range=(0, 30))
                    ax[i][j].set_title('%s %s' % (eyes[i], ' eye minimal distance'))
                    ax[i][j].set_xlabel('distance (cm)')
                    ax[i][j].set_ylabel('#counts')
                    ax[i][j].set_xticks([5, 15, 25])
                if j == 1:  # closest bodypart
                    distance_bodypart_matrix = np.array(
                        [np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][m2_bodypart]) for m2_bodypart in
                         m2_bodyparts])
                    closest_bodypart = [i for c in range(len(distance_bodypart_matrix[0])) for i, j in
                                        enumerate(distance_bodypart_matrix[:, c]) if
                                        j == np.nanmin(distance_bodypart_matrix[:, c])]
                    ax[i][j].hist(closest_bodypart, bins=8, range=(0, 8), align='left')
                    ax[i][j].set_title('%s %s' % (eyes[i], ' eye closest bodypart'))
                    ax[i][j].set_ylabel('#counts')
                    ax[i][j].set_xticklabels(labels=m2_bodyparts, rotation=45, rotation_mode="anchor")
        fig.tight_layout()
    plt.show(block=True)





# --------------------------------------------------------------------------------------------
# analysis 5: print a dictionary with the counts and and percentages of nan per m2_body_part
# --------------------------------------------------------------------------------------------



def cli_closed_eye_nan_counts(db_path,
                         mouse=None):
    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path,
                                                   video='rpi_camera_6',
                                                   min_likelihood=.99,
                                                   unit='cm')

        #add speed and averaged_speed to the tracking data for each bodypart
        tracking_data = func_speed_tracking_data(tracking_data)

        # load eye closure data
        eye_data = load_eye_closure_data(rec_path)

        #add averaged speed to eye data for each body part and each eye_timestamps
        eye_data = func_speed_eye_data(tracking_data,eye_data)

        #add the distance to m2 body parts to the tracking and eye data. And for the eye data also the closed eye speed.
        tracking_data,eye_data = func_abs_distance(tracking_data,eye_data)

        m2_bodyparts = sorted([k for k in tracking_data['body_parts'].keys() if k.startswith('m2')])

        nan_dict = {}
        for eye in ['left', 'right']:
            nan_dict[eye] = {}
            for bodypart in m2_bodyparts:
                nan_dict[eye][bodypart] = {}
                nan_dict[eye][bodypart]["#nan"] = len(
                    [c for c in np.concatenate(eye_data[eye]['closed_eye_distance'][bodypart]) if np.isnan(c)])
                nan_dict[eye][bodypart]["%nan"] = str(round(100 * (len(
                    [c for c in np.concatenate(eye_data[eye]['closed_eye_distance'][bodypart]) if np.isnan(c)]) / len(
                    np.concatenate(eye_data[eye]['closed_eye_distance'][bodypart]))))) + "%"
        pprint.pprint(nan_dict)








# ------------------------------------------------------------------
# analysis 6: make a video of the two mice with their whisker range
# ------------------------------------------------------------------


def closed_eye_tracking_data(tracking_data,eye_data):
    #First add to the tracking_data, the frames where eyes are closed
    tracking_data['eye_closed_interval'] = {}
    for eye in ['left','right']:
        eye_closed_timestamps_eye_data = np.concatenate([eye_data[eye]['timestamps'][c[0]:c[1] + 1] for c in eye_data[eye]['eye_closed_interval']])
        interp_f = interpolate.interp1d(tracking_data['timestamps'], list(range(len(tracking_data['timestamps']))),fill_value='extrapolate')
        eye_closed_interval_tracking_data = (np.round(interp_f(eye_closed_timestamps_eye_data))).astype(int)
        tracking_data['eye_closed_interval'][eye] = eye_closed_interval_tracking_data
    return tracking_data



def cli_whisker_video(db_path, output_path, mouse=None, camera_number=6, whisker_radius=1.5,test=False):  # circle radius in cm, if you set test=True, then only the first 1000 video frames will be loaded
    assert camera_number in [5, 6]

    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:
        print(mouse_id)
        print('Loading data')

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path, video='rpi_camera_' + str(camera_number),min_likelihood=.99, unit='cm')
        eye_data = load_eye_closure_data(rec_path)
        # add speed and averaged_speed to the tracking data for each bodypart
        tracking_data = func_speed_tracking_data(tracking_data)
        # load eye closure data
        # add averaged speed to eye data for each body part and each eye_timestamps
        eye_data = func_speed_eye_data(tracking_data, eye_data)
        # add the distance to m2 body parts to the tracking and eye data. And for the eye data also the closed eye speed.
        tracking_data, eye_data = func_abs_distance(tracking_data, eye_data)
        # add the intervals where the eyes are closed to the tracking_data (so with tracking_data timestamps)
        tracking_data = closed_eye_tracking_data(tracking_data, eye_data)

        # load the h5_file that belongs to mouse M3728 e.g. seen from camera 5 or 6 (camera_number)
        h5_file = pd.read_hdf(op.join(rec_path, recordings_mouse['camera_' + str(camera_number) + '_mice']))
        # Then make a lists of left eye, right eye, nose tip, (x,y,likelihood) list where the likelihood>0.99 for all parts.
        eye_left = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_left']))[:len(tracking_data['timestamps'])]
        eye_right = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_right']))[:len(tracking_data['timestamps'])]
        nose_tip = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_nose_tip']))[:len(tracking_data['timestamps'])]
        eye_mid = np.array([[(eye_left[c, 0] + eye_right[c, 0]) / 2, (eye_left[c, 1] + eye_right[c, 1]) / 2] for c in range(len(tracking_data['timestamps']))])
        head_mid = np.array([[(eye_mid[c, 0] + nose_tip[c, 0]) / 2, (eye_mid[c, 1] + nose_tip[c, 1]) / 2] for c in range(len(tracking_data['timestamps']))])

        # now look at the places where likelihood_threshold>0.99
        min_likelihood = 0.99
        indices_99 = np.array([i for i in range(len(eye_left[:, 2])) if eye_left[:, 2][i] > min_likelihood and eye_right[:, 2][i] > min_likelihood and nose_tip[:, 2][i] > min_likelihood])
        # look at the mean distance head_mid and eye_mid
        distance_99 = np.mean(np.sqrt((eye_mid[indices_99][:, 0] - head_mid[indices_99][:, 0]) ** 2 + (eye_mid[indices_99][:, 1] - head_mid[indices_99][:, 1]) ** 2))
        # calculate theta = dx/dy
        theta = np.arctan((eye_right[:, 0] - eye_left[:, 0]) / (eye_right[:, 1] - eye_left[:, 1]))
        # x_center = eye_mid + r*cos(theta). Here cos(theta) is taken absolute and there is looked for criteria when cos(theta) is positive or negative
        mid_estimate_x = eye_mid[:, 0] + distance_99 * np.array([np.abs(np.cos(theta[c])) if eye_left[c, 1] < eye_right[c, 1] else -np.abs(np.cos(theta[c])) for c in range(len(theta))])
        # y_center = eye_mid + r*sin(theta). Here sin(theta) is taken absolute and there is looked for criteria when sin(theta) is positive or negative
        mid_estimate_y = eye_mid[:, 1] + distance_99 * np.array([np.abs(np.sin(theta[c])) if eye_left[c, 0] > eye_right[c, 0] else -np.abs(np.sin(theta[c])) for c in range(len(theta))])

        # load the amount of pixels per cm
        pix_per_cm = helpers.get_pixels_per_centimeter(rec_path, video='rpi_camera_' + str(camera_number),marker1='corner_left_left', marker2='corner_lower_right')
        # look at the cm reference frame and where (x,y) for head center is
        mid_estimate_x_cm = mid_estimate_x / pix_per_cm
        mid_estimate_y_cm = mid_estimate_y / pix_per_cm
        r = whisker_radius * pix_per_cm

        #        for i in [8000]:
        #            data = reader.get_data(i)
        #            fig, ax = plt.subplots()
        #            ax.imshow(data)
        #            ax.scatter(eye_left[i, 0], eye_left[i, 1], color='blue', label='left eye')
        #            ax.scatter(eye_right[i, 0], eye_right[i, 1], color='green', label='right eye')
        #            #ax.scatter(nose_tip[i, 0], nose_tip[i, 1])
        #            #ax.scatter(head_mid[i, 0], head_mid[i, 1])
        #            ax.scatter(mid_estimate_x[i], mid_estimate_y[i], color='black', label='head center')
        #            ax.plot([eye_left[i, 0], eye_right[i, 0]], [eye_left[i, 1], eye_right[i, 1]])
        #            ax.plot([eye_mid[i, 0], mid_estimate_x[i]], [eye_mid[i, 1], mid_estimate_y[i]])
        #            #circle = plt.Circle((head_mid[i,0], head_mid[i,1]),r, color='red',fill=False,linewidth=1)
        #            circle = plt.Circle((mid_estimate_x[i], mid_estimate_y[i]), r, color='red', fill=False, linewidth=1,label='whisker range')
        #            ax.add_patch(circle)
        #            #ax.axis('scaled') #if you want to check if the lines are perpendicular (then also imshow off)
        #            ax.axis('off') #no axes
        #            fig.legend()
        #            fig.show()

        fps = 30  # frames per second of the output video
        bitrate = -1  # -1: automatically determine bitrate (= quality); use 2500 - 3000 for high-quality videos
        dpi = 150  # use 300 or more for higher quality
        writer = FFMpegWriter(fps=fps, metadata=dict(title='simple video example'), codec='libx264', bitrate=bitrate)
        reader = imageio.get_reader(op.join(rec_path, "rpi_camera_" + str(camera_number) + ".mp4"))
        plt.style.use('dark_background')  # use dark figure background for videos
        fig, ax = plt.subplots()  # create figure and axis
        ax.axis('off')  # no axes

        first_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # (height, width, rgb color channels)
        img = ax.imshow(first_frame)  # adding first figure

        whisker_circle = plt.Circle((mid_estimate_x[0], mid_estimate_y[0]), radius=r, color='tab:red', fill=False, linewidth=1,label='whisker range')
        ax.add_patch(whisker_circle)  # adding whisker circle

        eye_left_circle = plt.Circle((eye_left[0, 0], eye_left[0, 1]), radius=pix_per_cm*0.5, color='tab:green', label='eye')
        ax.add_patch(eye_left_circle)

        eye_right_circle = plt.Circle((eye_right[0, 0], eye_right[0, 1]), radius=pix_per_cm*0.5, color='tab:green')
        ax.add_patch(eye_right_circle)

        head_mid_circle = plt.Circle((mid_estimate_x[0], mid_estimate_y[0]), radius=pix_per_cm*0.5, color='tab:blue',label='head center')
        ax.add_patch(head_mid_circle)

        eyes_line = ax.plot((eye_left[0, 0], eye_right[0, 0]), (eye_left[0, 1], eye_right[0, 1]), '-', color='tab:orange', lw=2)[0]

        head_line = ax.plot((eye_mid[0, 0], mid_estimate_x[0]), (eye_mid[0, 1], mid_estimate_y[0]), '-', color='tab:orange', lw=2)[0]

        fig.legend()

        percentage = 0  # starting percentag for iteration
        video_range = len(eye_left)  # the total amount of video frames
        if test:
            video_range = 1000
        steps_of_showing = 100  # each hundredth (or what you want) is shown how much the video is processed
        hundredth = video_range / steps_of_showing

        print("Processing video")
        with writer.saving(fig, op.abspath(output_path)+'/whisker_video_'+ mouse_id+'.mp4', dpi=dpi):
            for i in range(video_range):

                if i > hundredth:
                    percentage += 1
                    print(str(int(percentage * 100 / steps_of_showing)) + '%')
                    hundredth += video_range / steps_of_showing

                data = reader.get_data(i)
                img.set_data(data)
                whisker_circle.set_center((mid_estimate_x[i], mid_estimate_y[i]))
                eye_left_circle.set_center((eye_left[i, 0], eye_left[i, 1]))
                eye_right_circle.set_center((eye_right[i, 0], eye_right[i, 1]))
                head_mid_circle.set_center((mid_estimate_x[i], mid_estimate_y[i]))
                eyes_line.set_data((eye_left[i, 0], eye_right[i, 0]), (eye_left[i, 1], eye_right[i, 1]))
                head_line.set_data((eye_mid[i, 0], mid_estimate_x[i]), (eye_mid[i, 1], mid_estimate_y[i]))

                writer.grab_frame()
        reader.close()





db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
output_path1 = "/Users/samsuidman/Desktop"
mouse1 = ['M4081']
cli_whisker_video(db_path1, output_path1, mouse=mouse1):  # circle radius in cm
