import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os.path
import time


t1 = time.time()

threshold = 0.99


def func_path(var_working_path): #returns an absolute working path as a variable
    var_abs_working_path = os.path.abspath(var_working_path)
    return var_abs_working_path

eyelid_left_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3729/together/cam3/rpi_camera_3DLC_resnet50_M3729_eyelidMar18shuffle1_500000.h5")
eyelid_right_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3729/together/cam4/rpi_camera_4DLC_resnet50_M3729_eyelidMar18shuffle1_500000.h5")
eyelid_left_npz_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3729/together/cam3/rpi_camera_3.npz")
eyelid_right_npz_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3729/together/cam4/rpi_camera_4.npz")


def func_h5_reader(var_path_to_h5_file): #reads a h5 file using the path to the file as a variable
    var_h5_file = pd.read_hdf(var_path_to_h5_file)
    return var_h5_file

eyelid_left_h5 = func_h5_reader(eyelid_left_h5_path)
eyelid_right_h5 = func_h5_reader(eyelid_right_h5_path)


def func_npz_reader(var_path_to_npz_file): #reads a npz file using the path to the file as a variable
    var_npz_file = np.load(var_path_to_npz_file)
    return var_npz_file

eyelid_left_npz = func_npz_reader(eyelid_left_npz_path)
eyelid_right_npz = func_npz_reader(eyelid_right_npz_path)


def func_likelihood_columns(var_mice_h5): #takes a h5_file with (x,y,likelihood)-data (including column names) and returns a list of the likelihood column names
    var_likelihood_columns = [] #make a list of likelihood column names (this is a vector)
    for i in var_mice_h5.columns:
        if i[2] == 'likelihood':
            var_likelihood_columns.append(i)
    return var_likelihood_columns

eyelid_left_likelihood_columns = func_likelihood_columns(eyelid_left_h5)
eyelid_right_likelihood_columns = func_likelihood_columns(eyelid_right_h5)



def func_likelihood(var_likelihood_columns,var_mice_h5): #takes a list h5 file with (x,y,likelihood)-data and a list of likelihood-column-names and returns a matrix of only the likelihood-data labeled by the column names
    var_likelihood = var_mice_h5[var_likelihood_columns] # make a list of likelihoodvalues per tracking object (so this is a matrix)
    return var_likelihood

eyelid_left_likelihood = func_likelihood(eyelid_left_likelihood_columns,eyelid_left_h5)
eyelid_right_likelihood = func_likelihood(eyelid_right_likelihood_columns,eyelid_right_h5)


def func_binary(var_likelihood,var_likelihood_columns,var_threshold): #Takes the total likelihood-matrix, the likelihood-columns-array and a threshold
    var_likelihood_binary = [] #Making a list for all bodyparts (len(list)=15) where if p<0.99 the value of list[i] becomes NaN and otherwise becomes 1+i (so that the lines are above eachother)
    k=0 #place for the first line
    for i in var_likelihood_columns: #going through each bodypart
        var_likelihood_binary_per_bodypart = [] #making a list per bodypart where [1,NaN,NaN,1,1,...]
        for j in var_likelihood[i]: #cloning the likelihood list of a bodypart [0.3338,0.9783,...]
            var_likelihood_binary_per_bodypart.append(j) # " " "
        var_likelihood_binary_per_bodypart = np.asarray(var_likelihood_binary_per_bodypart) #making type=array of it
        var_likelihood_binary_per_bodypart[var_likelihood_binary_per_bodypart<var_threshold] = np.NaN #setting p<0.99 to not a number
        var_likelihood_binary_per_bodypart[var_likelihood_binary_per_bodypart>var_threshold] = len(var_likelihood_columns)-k #setting p>0.99 to 15 (or 14,13,...,2,1) so that all lines can be plotted above eachother)
        var_likelihood_binary.append(var_likelihood_binary_per_bodypart) #adding the list of one bodypart to the bigger list
        k+=1 #setting k+1 so that the next bodypart comes at the line above/below the other
    return(var_likelihood_binary)

eyelid_left_likelihood_binary = func_binary(eyelid_left_likelihood,eyelid_left_likelihood_columns,threshold)
eyelid_right_likelihood_binary = func_binary(eyelid_right_likelihood,eyelid_right_likelihood_columns,threshold)



def func_high_likelihood(var_likelihood,var_likelihood_columns,threshold): #takes likelihood-matrix (it is matrix, because of multiple bodyparts), a likelihood-column-name array and a threshold and returns a matrix of low likelihoods and a matrix of low likelihood indices
    var_high_likelihood_values = []
    var_high_likelihood_index = []
    for i in var_likelihood_columns:
        var_high_likelihood_values_per_bodypart = var_likelihood[i].array[var_likelihood[i].array>threshold]
        var_high_likelihood_index_per_bodypart = var_likelihood[i].index[var_likelihood[i].array>threshold]
        var_high_likelihood_values.append(var_high_likelihood_values_per_bodypart)
        var_high_likelihood_index.append(var_high_likelihood_index_per_bodypart)
    return(var_high_likelihood_values,var_high_likelihood_index)

eyelid_left_high_likelihood_values,eyelid_left_high_likelihood_index = func_high_likelihood(eyelid_left_likelihood,eyelid_left_likelihood_columns,threshold)
eyelid_right_high_likelihood_values,eyelid_right_high_likelihood_index = func_high_likelihood(eyelid_right_likelihood,eyelid_right_likelihood_columns,threshold)





def func_high_likelihood_sequences(var_high_likelihood_values,var_high_likelihood_index,var_likelihood,var_likelihood_columns): #takes a list of low-likelihood-values, low-likelihood-indices and likelihood-column-names. Returns a list (15 elements for 15 bodyparts), where for each element there is a list with sequences of low-likelihood-frames. These for each sequences you can call (sequencelist.index) for an array of indices or (sequencelist.array) for an array of likelihoods.

    var_sequences_index = []
    for i in range(len(var_likelihood_columns)):
        var_sequences_index_per_bodypart = []  # making a list for one body part. Several lists of indices of subsequent high likelihoods are stored in this bigger list
        for var_first_sequence_frame in var_high_likelihood_index[i]:  # go through the list of all high likelihood frames
            if var_first_sequence_frame - 1 not in var_high_likelihood_index[i]:  # this is so that no lists are made that contain the same elements as another list
                var_second_variable = var_first_sequence_frame  # set a second variable to avoid conflict with the first one
                var_sequence = []  # making a list where subsequent high likelihood indices can be stored in
                var_sequence.append(var_second_variable)  # append the first high likelihood value to this list
                while var_second_variable + 1 in var_high_likelihood_index[i]:  # if the subsequent index of the high likelihood list is also in the high likelihood list, then look at this next index
                    var_second_variable += 1  # look at the next index
                    var_sequence.append(var_second_variable)  # add the next index also to the list
                var_sequences_index_per_bodypart.append(var_sequence)
        var_sequences_index.append(var_sequences_index_per_bodypart)

    var_sequences = []  # list where all reduced likelihoods for all bodyparts come in
    for j in range(len(var_likelihood_columns)):  # getting across all bodypart-indices (0-14)
        var_sequences_values_per_bodypart = []  # likelihood list per bodypart
        for k in var_sequences_index[j]:  # look at a specific list with subsequent indices in a bodypart list
            var_subsequent_likelihood = var_likelihood[var_likelihood_columns[j]][k]  # make a small list of likelihoods that match the small list of indices
            var_sequences_values_per_bodypart.append(var_subsequent_likelihood)  # add this small likelihood list to the list of one body part
        var_sequences.append(var_sequences_values_per_bodypart)  # add the one body part list to the list of all bodyparts

    return var_sequences

eyelid_left_high_sequences = func_high_likelihood_sequences(eyelid_left_high_likelihood_values,eyelid_left_high_likelihood_index,eyelid_left_likelihood,eyelid_left_likelihood_columns)
eyelid_right_high_sequences = func_high_likelihood_sequences(eyelid_right_high_likelihood_values,eyelid_right_high_likelihood_index,eyelid_right_likelihood,eyelid_right_likelihood_columns)







def func_low_likelihood(var_likelihood,var_likelihood_columns,threshold): #takes likelihood-matrix (it is matrix, because of multiple bodyparts), a likelihood-column-name array and a threshold and returns a matrix of low likelihoods and a matrix of low likelihood indices
    var_low_likelihood_values = []
    var_low_likelihood_index = []
    for i in var_likelihood_columns:
        var_low_likelihood_values_per_bodypart = var_likelihood[i].array[var_likelihood[i].array<threshold]
        var_low_likelihood_index_per_bodypart = var_likelihood[i].index[var_likelihood[i].array<threshold]
        var_low_likelihood_values.append(var_low_likelihood_values_per_bodypart)
        var_low_likelihood_index.append(var_low_likelihood_index_per_bodypart)
    return(var_low_likelihood_values,var_low_likelihood_index)

eyelid_left_low_likelihood_values,eyelid_left_low_likelihood_index = func_low_likelihood(eyelid_left_likelihood,eyelid_left_likelihood_columns,threshold)
eyelid_right_low_likelihood_values,eyelid_right_low_likelihood_index = func_low_likelihood(eyelid_right_likelihood,eyelid_right_likelihood_columns,threshold)





def func_low_likelihood_sequences(var_low_likelihood_values,var_low_likelihood_index,var_likelihood,var_likelihood_columns): #takes a list of low-likelihood-values, low-likelihood-indices and likelihood-column-names. Returns a list (15 elements for 15 bodyparts), where for each element there is a list with sequences of low-likelihood-frames. These for each sequences you can call (sequencelist.index) for an array of indices or (sequencelist.array) for an array of likelihoods.

    var_sequences_index = []
    for i in range(len(var_likelihood_columns)):
        var_sequences_index_per_bodypart = []  # making a list for one body part. Several lists of indices of subsequent low likelihoods are stored in this bigger list
        for var_first_sequence_frame in var_low_likelihood_index[i]:  # go through the list of all low likelihood frames
            if var_first_sequence_frame - 1 not in var_low_likelihood_index[i]:  # this is so that no lists are made that contain the same elements as another list
                var_second_variable = var_first_sequence_frame  # set a second variable to avoid conflict with the first one
                var_sequence = []  # making a list where subsequent low likelihood indices can be stored in
                var_sequence.append(var_second_variable)  # append the first low likelihood value to this list
                while var_second_variable + 1 in var_low_likelihood_index[i]:  # if the subsequent index of the low likelihood list is also in the low likelihood list, then look at this next index
                    var_second_variable += 1  # look at the next index
                    var_sequence.append(var_second_variable)  # add the next index also to the list
                var_sequences_index_per_bodypart.append(var_sequence)
        var_sequences_index.append(var_sequences_index_per_bodypart)

    var_sequences = []  # list where all reduced likelihoods for all bodyparts come in
    for j in range(len(var_likelihood_columns)):  # getting across all bodypart-indices (0-14)
        var_sequences_values_per_bodypart = []  # likelihood list per bodypart
        for k in var_sequences_index[j]:  # look at a specific list with subsequent indices in a bodypart list
            var_subsequent_likelihood = var_likelihood[var_likelihood_columns[j]][k]  # make a small list of likelihoods that match the small list of indices
            var_sequences_values_per_bodypart.append(var_subsequent_likelihood)  # add this small likelihood list to the list of one body part
        var_sequences.append(var_sequences_values_per_bodypart)  # add the one body part list to the list of all bodyparts

    return var_sequences

eyelid_left_low_sequences = func_low_likelihood_sequences(eyelid_left_low_likelihood_values,eyelid_left_low_likelihood_index,eyelid_left_likelihood,eyelid_left_likelihood_columns)
eyelid_right_high_sequences = func_low_likelihood_sequences(eyelid_right_low_likelihood_values,eyelid_right_low_likelihood_index,eyelid_right_likelihood,eyelid_right_likelihood_columns)




fig_left, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
fig_left.suptitle('eyelid_left_M7329_M7328')
ax1.plot(eyelid_left_likelihood[eyelid_left_likelihood_columns[2]])
ax1.set_title('{}'.format(eyelid_left_likelihood_columns[2][1]))
ax2.plot(eyelid_left_likelihood[eyelid_left_likelihood_columns[3]])
ax2.set_title('{}'.format(eyelid_left_likelihood_columns[3][1]))
ax3.plot(eyelid_left_likelihood[eyelid_left_likelihood_columns[4]])
ax3.set_title('{}'.format(eyelid_left_likelihood_columns[4][1]))
fig_left.show()



fig_right, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
fig_right.suptitle('eyelid_right_M7329_M7328')
ax1.plot(eyelid_right_likelihood[eyelid_right_likelihood_columns[2]])
ax1.set_title('{}'.format(eyelid_right_likelihood_columns[2][1]))
ax2.plot(eyelid_right_likelihood[eyelid_right_likelihood_columns[3]])
ax2.set_title('{}'.format(eyelid_right_likelihood_columns[3][1]))
ax3.plot(eyelid_right_likelihood[eyelid_right_likelihood_columns[4]])
ax3.set_title('{}'.format(eyelid_right_likelihood_columns[4][1]))
fig_right.show()






def func_compressed_sequences(var_high_sequences,var_continued_frames): #takes 2 things: 1) a big list with 15 elements (each bodypart), each with a list of sequences (sequences are pandas.series.Series) 2) the maximum amount of frames (plus 1) that can be between to sequences when merging sequences together. The function returns a big list containing lists with sequences that are merged together.
    var_high_sequences_compressed = []
    for var_high_sequences_per_bodypart in var_high_sequences:
        var_index_list = [] #making a big list, where all the indices are coming from the lists that should be merged together
        var_double_indices = [] #here is a list that can be used to see if a certain list(-index) isn't already added to another list
        for i in range(len(var_high_sequences_per_bodypart)): #iterate over the sequences (the indices i of them)
            if i not in var_double_indices: #check if i not in double list
                var_index_list_temp = [] #make a temporary index list, where indices of sequences that have to be merged together come into
                var_index_list_temp.append(i) # add the (first) index i
                for j in range(len(var_high_sequences_per_bodypart)): #make a new iteration starting from i
                    if i+j < len(var_high_sequences_per_bodypart)-1: #check if new iteration is not out of range
                        var_difference = var_high_sequences_per_bodypart[i+j+1].index[0]-var_high_sequences_per_bodypart[i+j].index[len(var_high_sequences_per_bodypart[i+j])-1] #set variable of difference between lists
                        if var_difference < var_continued_frames: #check if the difference is not bigger than the difference that you want
                            var_index_list_temp.append(i+j+1) #add the index of a list that should be merged together to the temporary index list
                            var_double_indices.append(i+j+1) #make sure that this index would not be added to another list too
                    if var_difference >= var_continued_frames: #if the difference is bigger than the difference you want, then from here on you want to begin a new big list with merged smaller lists
                        break #so you break
                var_index_list.append(var_index_list_temp) #you add the list of merged lists (the indices) to the big list
        var_high_sequences_compressed_per_bodypart = [] #you make a list where you add the actual values of the panda.series.Series(-indices) to
        for i in var_index_list: #you iterate over the big list that contains small lists of compressed sequence-lists
            var_high_sequences_compressed_per_bodypart_temp = [] #you look at one of those compressed (merged) sequences
            for j in i: #you look at an index value of one sequence
                var_high_sequences_compressed_per_bodypart_temp.extend(var_high_sequences_per_bodypart[j].index) #you merge the sequences that should be together
            var_high_sequences_compressed_per_bodypart.append(var_high_sequences_compressed_per_bodypart_temp) #you add this list of compressed sequences to the big list
        var_high_sequences_compressed.append(var_high_sequences_compressed_per_bodypart)
    return(var_high_sequences_compressed) #you give back a big list that contains lists of compressed sequences

eyelid_left_compressed_sequences = func_compressed_sequences(eyelid_left_high_sequences,200)
eyelid_right_closed_eye_sequences = func_compressed_sequences(eyelid_right_high_sequences,200)

...:


def func_video(var_video_path,var_compressed_sequences_per_bodypart):  # takes a path to a video and returns the metadata and the amount of frames of the video (as tuple)
    with imageio.get_reader(var_video_path) as var_video:  # with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
        var_video_meta_data = var_video.get_meta_data()  # this contains the metadata, such as  fps (frames per second), duration, etc.
        var_video_frames_count = var_video.count_frames()  # counting the amount of frames (=19498)
        var_video_dataframes = []
        for var_sequence in var_compressed_sequences_per_bodypart:

            var_video.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 19497
    return var_video_meta_data, var_video_frames_count






def func_video_writer(var_video_path,var_black_path,var_compressed_sequences_per_bodypart): #input is the video path and the compressed sequences FOR ONE BODYPART!!!!! (so "eyelid_left_compressed_sequences[4]" for the closed eyelid)
    var_output_path = os.path.splitext(var_video_path)[0] + '_converted_video' + '.mp4'
    var_reader = imageio.get_reader(var_video_path)

    var_test_frame = var_reader.get_data(0)
    var_black_frame = np.zeros([var_test_frame.shape[0], var_test_frame.shape[1], var_test_frame.shape[2]], dtype=np.uint8)
    var_black_frame.fill(255)  # or img[:] = 255

    var_fps = var_reader.get_meta_data()['fps']
    var_writer = imageio.get_writer(var_output_path,fps=var_fps)

    for var_sequence in var_compressed_sequences_per_bodypart:
        for var_index in var_sequence:
            var_frame = var_reader.get_data(var_index)
            var_writer.append_data(var_frame)
        for i in range(int(var_fps/3)):
            var_writer.append_data(var_black_frame)
#func_video_writer('/Users/samsuidman/Desktop/video_test_map/rpi_camera_4.mp4','/Users/samsuidman/Downloads/zwart_foto.jpg',eyelid_right_compressed_sequences[4])



def func_smooth_sequences(var_compressed_sequences):
    var_smooth_sequences = []
    for var_compressed_sequences_per_bodypart in var_compressed_sequences: #look at a specific bodypart
        var_smooth_sequences_per_bodypart = []
        for var_sequence in var_compressed_sequences_per_bodypart:
            var_first_sequence_index = var_sequence[0]
            var_last_sequence_index = var_sequence[len(var_sequence)-1]
            var_smooth_single_sequence = list(range(var_first_sequence_index,var_last_sequence_index+1))
            var_smooth_sequences_per_bodypart.append(var_smooth_single_sequence)
        var_smooth_sequences.append(var_smooth_sequences_per_bodypart)
    return var_smooth_sequences

eyelid_left_smooth_sequences = func_smooth_sequences(eyelid_left_compressed_sequences)
eyelid_right_smooth_sequences = func_smooth_sequences(eyelid_right_compressed_sequences)





def func_x_y_likelihood_columns(var_mice_h5,var_x_y_likelihood): #takes 2 things: 1) a h5_file with (x,y,likelihood)-data (including column names) 2) a string that says "x","y","likelihood" (what kind of column names you want) and returns a list of the x, the y or the likelihood column names
    var_x_y_likelihood_columns = [] #make a list of likelihood column names (this is a vector)
    for i in var_mice_h5.columns:
        if i[2] == var_x_y_likelihood:
            var_x_y_likelihood_columns.append(i)
    return var_x_y_likelihood_columns

mice_cam5_likelihood_columns = func_x_y_likelihood_columns(mice_cam5_h5,"likelihood")


def func_x_y_likelihood(var_x_y_likelihood_columns,var_mice_h5): #takes a list h5 file with (x,y,likelihood)-data and a list of x- ,y- or likelihood-column-names and returns a matrix of only the data labeled by the column names
    var_x_y_likelihood = var_mice_h5[var_x_y_likelihood_columns] # make a list of x-, y- or likelihood-values per tracking object (so this is a matrix)
    return var_x_y_likelihood

mice_cam5_likelihood = func_x_y_likelihood(mice_cam5_likelihood_columns,mice_cam5_h5)





breedte = 480
hoogte = 640

def func_four_images(var_data_up_left,var_data_up_right,var_data_down_left,var_data_down_right): #Takes the data (reader.get_data(i)) of four same-size-pictures and returns the data (same data format) of the four pictures combined.
    var_data_up = [] #combining the up left and up right pictures
    for var_i in range(len(var_data_up_left)): #look at each horizontal row of pixels in the image
        var_horizantal_pixels_up = np.concatenate((var_data_up_left[var_i], var_data_up_right[var_i])) #combining the 2 horizontal rows from the pictures
        var_data_up.append(var_horizantal_pixels_up) #append each row to the list. Each item in this list is a horizontal row, so this list is VERTICAL
    var_data_up = np.asarray(var_data_up) #make a ndarray of the list, so that the writer function can write a new picture

    var_data_down = [] #combining the down left and down right pictures
    for var_j in range(len(var_data_down_left)): #look at each horizontal row of pixels in the image
        var_horizontal_pixels_down = np.concatenate((var_data_down_left[var_j], var_data_down_right[var_j])) #combining the 2 horizontal rows from the pictures
        var_data_down.append(var_horizontal_pixels_down) #append each row to the list. Each item in this list is a horizontal row, so this list is VERTICAL
    var_data_down = np.asarray(var_data_down) #make a ndarray of the list, so that the writer function can write a new picture

    var_data = np.concatenate((var_data_up, var_data_down)) #here the the two arrays are combined to a tuple, and then the tuple will form an array. This way the combined two arrays still form an array.
    return var_data #return the data of the new picture with four pictures combined


reader_right = imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam3/raw_video/rpi_camera_3.mp4")
data0_right = reader_right.get_data(0)
fps_right = reader_right.get_meta_data()['fps']

reader_left = imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam4/raw_video/rpi_camera_4.mp4")
data0_left = reader_left.get_data(0)
fps_left = reader_left.get_meta_data()['fps']

reader_cam5 = imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam5/raw_video/rpi_camera_5.mp4")
data0_cam5 = reader_cam5.get_data(0)
fps_cam5 = reader_cam5.get_meta_data()['fps']

reader_cam6 = imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam6/raw_video/rpi_camera_6.mp4")
data0_cam6 = reader_cam6.get_data(0)
fps_cam6 = reader_cam6.get_meta_data()['fps']

data = func_four_images(data0_right,data0_left,data0_cam5,data0_cam6)
writer = imageio.get_writer("/Users/samsuidman/Desktop/likelihood_figures/four_images.png")
writer.append_data(data)








# -------------------------------------------------------------------------------------------------------------------
#  Making a whisker radius around the mouse head center. This can be copy pasted under eye_closure_analysis_test.py
# -------------------------------------------------------------------------------------------------------------------

def closed_eye_tracking_data(tracking_data,eye_data):
    #First add to the tracking_data, the frames where eyes are closed
    tracking_data['eye_closed_interval'] = {}
    for eye in ['left','right']:
        eye_closed_timestamps_eye_data = np.concatenate([eye_data[eye]['timestamps'][c[0]:c[1] + 1] for c in eye_data[eye]['eye_closed_interval']])
        interp_f = interpolate.interp1d(tracking_data['timestamps'], list(range(len(tracking_data['timestamps']))))
        eye_closed_interval_tracking_data = (np.round(interp_f(eye_closed_timestamps_eye_data))).astype(int)
        tracking_data['eye_closed_interval'][eye] = eye_closed_interval_tracking_data
    return tracking_data


def whisker_model(db_path,mouse=None,camera_number=6,circle_radius=1.5): #circle radius in cm
    assert camera_number in [5,6]

    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

        recordings_mouse = get_recordings_mouse(mouse_id)
        rec_path = op.join(db_path, recordings_mouse['session'], recordings_mouse['interaction'])

        # load tracking data (in egocentric reference frame)
        tracking_data = helpers.load_tracking_data(rec_path,video='rpi_camera_'+str(camera_number),min_likelihood=.99,unit='cm')
        eye_data = load_eye_closure_data(rec_path)

        #add speed and averaged_speed to the tracking data for each bodypart
        tracking_data = func_speed_tracking_data(tracking_data)
        # load eye closure data
        #add averaged speed to eye data for each body part and each eye_timestamps
        eye_data = func_speed_eye_data(tracking_data,eye_data)
        #add the distance to m2 body parts to the tracking and eye data. And for the eye data also the closed eye speed.
        tracking_data,eye_data = func_abs_distance(tracking_data,eye_data)
        #add the intervals where the eyes are closed to the tracking_data (so with tracking_data timestamps)
        tracking_data = closed_eye_tracking_data(tracking_data,eye_data)

        #load the h5_file that belongs to mouse M3728 e.g. seen from camera 5 or 6 (camera_number)
        h5_file = pd.read_hdf(op.join(rec_path, recordings_mouse['camera_'+str(camera_number)+'_mice']))
        # Then make a lists of left eye, right eye, nose tip, (x,y,likelihood) list where the likelihood>0.99 for all parts.
        eye_left = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_left']))[:len(tracking_data['timestamps'])]
        eye_right = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_right']))[:len(tracking_data['timestamps'])]
        nose_tip = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_nose_tip']))[:len(tracking_data['timestamps'])]
        eye_mid = np.array([[(eye_left[c, 0] + eye_right[c, 0]) / 2, (eye_left[c, 1] + eye_right[c, 1]) / 2] for c in range(len(tracking_data['timestamps']))])
        head_mid = np.array([[(eye_mid[c, 0] + nose_tip[c, 0]) / 2, (eye_mid[c, 1] + nose_tip[c, 1]) / 2] for c in range(len(tracking_data['timestamps']))])
        #now look at the places where likelihood_threshold>0.99
        min_likelihood=0.99
        indices_99 = np.array([i for i in range(len(eye_left[:, 2])) if eye_left[:, 2][i] > min_likelihood and eye_right[:, 2][i] > min_likelihood and nose_tip[:, 2][i] > min_likelihood])
#        eye_left_99 = eye_left[indices_99] #and for head_mid etc

        #now try to make look at the pictures with a center at a certain place
        pix_per_cm = helpers.get_pixels_per_centimeter(rec_path,video='rpi_camera_'+str(camera_number),marker1='corner_left_left',  marker2='corner_lower_right')
        r = circle_radius*pix_per_cm
        reader = imageio.get_reader(op.join(rec_path,"rpi_camera_"+str(camera_number)+".mp4"))
        for i in [indices_99[0], indices_99[100], indices_99[200], indices_99[300], indices_99[400], indices_99[500]]:
            data = reader.get_data(i)
            fig, ax = plt.subplots()
            ax.imshow(data)
            #ax.scatter(eye_left[i, 0], eye_left[i, 1])
            #ax.scatter(eye_right[i, 0], eye_right[i, 1])
            #ax.scatter(nose_tip[i, 0], nose_tip[i, 1])
            #ax.scatter(head_mid[i, 0], head_mid[i, 1])
            circle = plt.Circle((head_mid[i,0], head_mid[i,1]),r, color='red',fill=False,linewidth=1)
            ax.add_patch(circle)
            fig.show()
        reader.close()

    return tracking_data,eye_data


db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M4081']
tracking_data,eye_data = whisker_model(db_path1,mouse1)
