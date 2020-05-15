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
                  'eyelid_right': 'rpi_camera_4DLC_resnet50_M3728_eyelidMar25shuffle1_1030000.h5'},
        'M3729': {'session': 'M3729/2019_07_08/social_interaction',
                  'interaction': '2019-07-08_12-15-38_Silence_box_no_enclosure_M3728',
                  'alone': '2019-07-08_12-05-16_Silence_box_no_enclosure',
                  'eyelid_left': 'rpi_camera_3DLC_resnet50_M3729_eyelidMar18shuffle1_500000.h5',
                  'eyelid_right': 'rpi_camera_4DLC_resnet50_M3729_eyelidMar18shuffle1_500000.h5'},
        'M4081': {'session': 'M4081/2019_08_07/social_interaction',
                  'interaction': '2019-08-07_15-59-38_Silence_box_no_enclosure_M3729',
                  'alone': '2019-08-07_15-45-22_Silence_box_no_enclosure',
                  'eyelid_left': 'rpi_camera_3DLC_resnet50_M4081_eyelidMar19shuffle1_1030000.h5',
                  'eyelid_right': 'rpi_camera_4DLC_resnet50_M4081_eyelidMar19shuffle1_1030000.h5'}
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


