import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os.path
import time
from scipy.interpolate import interp1d

t1 = time.time()





threshold = 0.99 #the threshold where you tell that the tracking is done well
time_between_frames = 30 #the amount of frames that can be between two eye closure events, before you say it is one event. (fps is around 60)





def func_path(var_working_path): #returns an absolute working path as a variable
    var_abs_working_path = os.path.abspath(var_working_path)
    return var_abs_working_path

mice_cam5_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam5/mice/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5")
mice_cam5_npz_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam5/rpi_camera_5.npz")
mice_cam5_video_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam5/tracked_video/mice/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000_labeled.mp4")
mice_cam6_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam6/mice/rpi_camera_6DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5")
mice_cam6_npz_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam6/rpi_camera_6.npz")
mice_cam6_video_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam6/tracked_video/mice/rpi_camera_6DLC_resnet50_M3728_miceFeb14shuffle1_1030000_labeled.mp4")
eyelid_left_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam3/rpi_camera_3DLC_resnet50_M3728_eyelidMar18shuffle1_200000.h5")
eyelid_left_npz_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam3/rpi_camera_3.npz")
eyelid_left_video_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam3/raw_video/rpi_camera_3.mp4")
eyelid_right_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam4/rpi_camera_4DLC_resnet50_M3728_eyelidMar25shuffle1_1030000.h5")
eyelid_right_npz_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam4/rpi_camera_4.npz")
eyelid_right_video_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam4/raw_video/rpi_camera_4.mp4")
figure_save_path = func_path("/Users/samsuidman/Desktop/likelihood_figures")




#reading the tracking files (h5), these variables now contain a matrix, 3 elements per
#bodypart (x,y,likelihood). There are 10 bodyparts, so in total there are 30 elements.
#Each element got almost 20000 values, which are coordinate-values (for x and y) or
#likelihoodvalues
def func_h5_reader(var_path_to_h5_file): #reads a h5 file using the path to the file as a variable
    var_h5_file = pd.read_hdf(var_path_to_h5_file)
    return var_h5_file

mice_cam5_h5 = func_h5_reader(mice_cam5_h5_path)
mice_cam6_h5 = func_h5_reader(mice_cam6_h5_path)
eyelid_left_h5 = func_h5_reader(eyelid_left_h5_path)
eyelid_right_h5 = func_h5_reader(eyelid_right_h5_path)




def func_npz_reader(var_path_to_npz_file): #for information type "mice_cam5_npz.files"
    var_npz_file = np.load(var_path_to_npz_file)
    return var_npz_file

mice_cam5_npz = func_npz_reader(mice_cam5_npz_path)
mice_cam6_npz = func_npz_reader(mice_cam6_npz_path)
eyelid_left_npz = func_npz_reader(eyelid_left_npz_path)
eyelid_right_npz = func_npz_reader(eyelid_right_npz_path)




def func_timestamps(var_mice_npz): #takes a npz-file and returns timestamps array
    var_mice_timestamps = var_mice_npz["timestamps"]
    return var_mice_timestamps

mice_cam5_timestamps = func_timestamps(mice_cam5_npz)
mice_cam6_timestamps = func_timestamps(mice_cam6_npz)
eyelid_left_timestamps = func_timestamps(eyelid_left_npz)
eyelid_right_timestamps = func_timestamps(eyelid_right_npz)





def func_x_y_likelihood_columns(var_mice_h5,var_x_y_likelihood): #takes 2 things: 1) a h5_file with (x,y,likelihood)-data (including column names) 2) a string that says "x","y","likelihood" (what kind of column names you want) and returns a list of the x, the y or the likelihood column names
    var_x_y_likelihood_columns = [i for i in var_mice_h5 if i[2] == var_x_y_likelihood] #make a list of likelihood column names (this is a vector)
    return var_x_y_likelihood_columns

mice_cam5_likelihood_columns = func_x_y_likelihood_columns(mice_cam5_h5,"likelihood")
mice_cam6_likelihood_columns = func_x_y_likelihood_columns(mice_cam6_h5,"likelihood")
eyelid_left_likelihood_columns = func_x_y_likelihood_columns(eyelid_left_h5,"likelihood")
eyelid_right_likelihood_columns = func_x_y_likelihood_columns(eyelid_right_h5,"likelihood")
mice_cam5_x_columns = func_x_y_likelihood_columns(mice_cam5_h5,"x")
mice_cam6_x_columns = func_x_y_likelihood_columns(mice_cam6_h5,"x")
eyelid_left_x_columns = func_x_y_likelihood_columns(eyelid_left_h5,"x")
eyelid_right_x_columns = func_x_y_likelihood_columns(eyelid_right_h5,"x")
mice_cam5_y_columns = func_x_y_likelihood_columns(mice_cam5_h5,"y")
mice_cam6_y_columns = func_x_y_likelihood_columns(mice_cam6_h5,"y")
eyelid_left_y_columns = func_x_y_likelihood_columns(eyelid_left_h5,"y")
eyelid_right_y_columns = func_x_y_likelihood_columns(eyelid_right_h5,"y")





def func_x_y_likelihood(var_x_y_likelihood_columns,var_mice_h5): #takes a list h5 file with (x,y,likelihood)-data and a list of x- ,y- or likelihood-column-names and returns a matrix of only the data labeled by the column names
    var_x_y_likelihood = var_mice_h5[var_x_y_likelihood_columns] # make a list of x-, y- or likelihood-values per tracking object (so this is a matrix)
    return var_x_y_likelihood

mice_cam5_likelihood = func_x_y_likelihood(mice_cam5_likelihood_columns,mice_cam5_h5)
mice_cam6_likelihood = func_x_y_likelihood(mice_cam6_likelihood_columns,mice_cam6_h5)
eyelid_left_likelihood = func_x_y_likelihood(eyelid_left_likelihood_columns,eyelid_left_h5)
eyelid_right_likelihood = func_x_y_likelihood(eyelid_right_likelihood_columns,eyelid_right_h5)
mice_cam5_x = func_x_y_likelihood(mice_cam5_x_columns,mice_cam5_h5)
mice_cam6_x = func_x_y_likelihood(mice_cam6_x_columns,mice_cam6_h5)
eyelid_left_x = func_x_y_likelihood(eyelid_left_x_columns,eyelid_left_h5)
eyelid_right_x = func_x_y_likelihood(eyelid_right_x_columns,eyelid_right_h5)
mice_cam5_y = func_x_y_likelihood(mice_cam5_y_columns,mice_cam5_h5)
mice_cam6_y = func_x_y_likelihood(mice_cam6_y_columns,mice_cam6_h5)
eyelid_left_y = func_x_y_likelihood(eyelid_left_y_columns,eyelid_left_h5)
eyelid_right_y = func_x_y_likelihood(eyelid_right_y_columns,eyelid_right_h5)





def func_video(var_video_path): #takes a path to a video and returns the metadata and the amount of frames of the video (as tuple)
    with imageio.get_reader(var_video_path) as var_video:     #with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
        var_video_meta_data = var_video.get_meta_data() #this contains the metadata, such as  fps (frames per second), duration, etc.
        var_video_frames = var_video.count_frames() #counting the amount of frames (=19498)
    #    var_video_dataframe_2364 = var_video.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 19497
    return var_video_meta_data, var_video_frames

mice_cam5_video_info = func_video(mice_cam5_video_path)
mice_cam6_video_info = func_video(mice_cam6_video_path)
eyelid_left_video_info = func_video(eyelid_left_video_path)
eyelid_right_video_info = func_video(eyelid_right_video_path)





def func_low_likelihood(var_likelihood,var_likelihood_columns,threshold): #takes likelihood-matrix (it is matrix, because of multiple bodyparts), a likelihood-column-name array and a threshold and returns a matrix of low likelihoods and a matrix of low likelihood indices
    var_low_likelihood_values = [var_likelihood[i].array[var_likelihood[i].array<threshold] for i in var_likelihood_columns]
    var_low_likelihood_index = [var_likelihood[j].index[var_likelihood[j].array<threshold] for j in var_likelihood_columns]
    return(var_low_likelihood_values,var_low_likelihood_index)

mice_cam5_low_likelihood_values,mice_cam5_low_likelihood_index = func_low_likelihood(mice_cam5_likelihood,mice_cam5_likelihood_columns,threshold)
mice_cam6_low_likelihood_values,mice_cam6_low_likelihood_index = func_low_likelihood(mice_cam6_likelihood,mice_cam6_likelihood_columns,threshold)
eyelid_left_low_likelihood_values,eyelid_left_low_likelihood_index = func_low_likelihood(eyelid_left_likelihood,eyelid_left_likelihood_columns,threshold)
eyelid_right_low_likelihood_values,eyelid_right_low_likelihood_index = func_low_likelihood(eyelid_right_likelihood,eyelid_right_likelihood_columns,threshold)





def func_high_likelihood(var_likelihood,var_likelihood_columns,threshold): #takes likelihood-matrix (it is matrix, because of multiple bodyparts), a likelihood-column-name array and a threshold and returns a matrix of low likelihoods and a matrix of low likelihood indices
    var_high_likelihood_values = [var_likelihood[i].array[var_likelihood[i].array>threshold] for i in var_likelihood_columns]
    var_high_likelihood_index = [var_likelihood[j].index[var_likelihood[j].array>threshold] for j in var_likelihood_columns]
    return(var_high_likelihood_values,var_high_likelihood_index)

mice_cam5_high_likelihood_values,mice_cam5_high_likelihood_index = func_high_likelihood(mice_cam5_likelihood,mice_cam5_likelihood_columns,threshold)
mice_cam6_high_likelihood_values,mice_cam6_high_likelihood_index = func_high_likelihood(mice_cam6_likelihood,mice_cam6_likelihood_columns,threshold)
eyelid_left_high_likelihood_values,eyelid_left_high_likelihood_index = func_high_likelihood(eyelid_left_likelihood,eyelid_left_likelihood_columns,threshold)
eyelid_right_high_likelihood_values,eyelid_right_high_likelihood_index = func_high_likelihood(eyelid_right_likelihood,eyelid_right_likelihood_columns,threshold)




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

mice_cam5_low_sequences = func_low_likelihood_sequences(mice_cam5_low_likelihood_values,mice_cam5_low_likelihood_index,mice_cam5_likelihood,mice_cam5_likelihood_columns)
mice_cam6_low_sequences = func_low_likelihood_sequences(mice_cam6_low_likelihood_values,mice_cam6_low_likelihood_index,mice_cam6_likelihood,mice_cam6_likelihood_columns)
eyelid_left_low_sequences = func_low_likelihood_sequences(eyelid_left_low_likelihood_values,eyelid_left_low_likelihood_index,eyelid_left_likelihood,eyelid_left_likelihood_columns)
eyelid_right_low_sequences = func_low_likelihood_sequences(eyelid_right_low_likelihood_values,eyelid_right_low_likelihood_index,eyelid_right_likelihood,eyelid_right_likelihood_columns)





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

mice_cam5_high_sequences = func_high_likelihood_sequences(mice_cam5_high_likelihood_values,mice_cam5_high_likelihood_index,mice_cam5_likelihood,mice_cam5_likelihood_columns)
mice_cam6_high_sequences = func_high_likelihood_sequences(mice_cam6_high_likelihood_values,mice_cam6_high_likelihood_index,mice_cam6_likelihood,mice_cam6_likelihood_columns)
eyelid_left_high_sequences = func_high_likelihood_sequences(eyelid_left_high_likelihood_values,eyelid_left_high_likelihood_index,eyelid_left_likelihood,eyelid_left_likelihood_columns)
eyelid_right_high_sequences = func_high_likelihood_sequences(eyelid_right_high_likelihood_values,eyelid_right_high_likelihood_index,eyelid_right_likelihood,eyelid_right_likelihood_columns)





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

mice_cam5_compressed_sequences = func_compressed_sequences(mice_cam5_high_sequences,time_between_frames)
mice_cam6_compressed_sequences = func_compressed_sequences(mice_cam6_high_sequences,time_between_frames)
eyelid_left_compressed_sequences = func_compressed_sequences(eyelid_left_high_sequences,time_between_frames)
eyelid_right_compressed_sequences = func_compressed_sequences(eyelid_right_high_sequences,time_between_frames)





def func_frame_of_lowest_likelihood(var_sequences): #Takes a list matrix (because of 15 bodyparts) of sequences and returns two matrices (index and likelihood) of the frame with the lowest likelihood per sequence.
    var_lowest_likelihood_index = [] #Making a list (within it a list) from the indices where a sequence has a minimal likelihood
    var_lowest_likelihood_values = [] #Making the list with corresponding likelihoods
    for j in var_sequences: #look at the 15 bodyparts
        var_lowest_likelihood_index_per_bodypart = [] #look at index per bodypart
        var_lowest_likelihood_values_per_bodypart = [] #look at the likelihood per bodypart
        for i in j: #look at the list-sequences per bodypart
            var_index_min = i.idxmin() #look at the index corresponding to the minimal likelihood in a sequence
            var_likelihood_min = min(i.array) #looking at the minimal likelihood of a sequence
            var_lowest_likelihood_index_per_bodypart.append(var_index_min) #add the index to the index_list per bodypart
            var_lowest_likelihood_values_per_bodypart.append(var_likelihood_min) #add the likelihood to the likelihood_list per bodypart
        var_lowest_likelihood_index.append(var_lowest_likelihood_index_per_bodypart) #add each index bodypart list to the big index list
        var_lowest_likelihood_values.append(var_lowest_likelihood_values_per_bodypart) #add each likelihood bodypart list ot the big likelihood bodypart list
    return(var_lowest_likelihood_index,var_lowest_likelihood_values)

mice_cam5_lowest_likelihood_index, mice_cam5_lowest_likelihood_values = func_frame_of_lowest_likelihood(mice_cam5_low_sequences)
mice_cam6_lowest_likelihood_index, mice_cam6_lowest_likelihood_values = func_frame_of_lowest_likelihood(mice_cam6_low_sequences)
eyelid_left_lowest_likelihood_index, eyelid_left_lowest_likelihood_values = func_frame_of_lowest_likelihood(eyelid_left_low_sequences)
eyelid_right_lowest_likelihood_index, eyelid_right_lowest_likelihood_values = func_frame_of_lowest_likelihood(eyelid_right_low_sequences)





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

mice_cam5_likelihood_binary = func_binary(mice_cam5_likelihood,mice_cam5_likelihood_columns,threshold)
mice_cam6_likelihood_binary = func_binary(mice_cam6_likelihood,mice_cam6_likelihood_columns,threshold)
eyelid_left_likelihood_binary = func_binary(eyelid_left_likelihood,eyelid_left_likelihood_columns,threshold)
eyelid_right_likelihood_binary = func_binary(eyelid_right_likelihood,eyelid_right_likelihood_columns,threshold)





def func_plot_show(var_likelihood_binary,var_likelihood,var_likelihood_columns):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(10.0,4.8))
    for j in range(len(var_likelihood_columns)):  # going through each body part
        ax2.plot(var_likelihood_binary[j],label="{}".format(var_likelihood_columns[j][1]))  # plot the line of the binary likelihood
        ax3.plot(list(var_likelihood[var_likelihood_columns[j]].index / len(var_likelihood[var_likelihood_columns[j]])),var_likelihood_binary[j],label="{}".format(var_likelihood_columns[j][1]))  # plot the line of the binary likelihood
        ax1.plot([0, 1], [0, 0], label="{}".format(var_likelihood_columns[j][1]))
        ax1.legend()  # making the legend
    fig.show()  # saving the picture at high quality

#func_plot_show(mice_cam5_likelihood_binary,mice_cam5_likelihood,mice_cam5_likelihood_columns)
#func_plot_show(mice_cam6_likelihood_binary,mice_cam6_likelihood,mice_cam6_likelihood_columns)
#func_plot_show(eyelid_left_likelihood_binary,eyelid_left_likelihood,eyelid_left_likelihood_columns)
#func_plot_show(eyelid_right_likelihood_binary,eyelid_right_likelihood,eyelid_right_likelihood_columns)





def func_plot_save(var_likelihood_binary,var_likelihood,var_likelihood_columns,var_figure_save_path):
    var_fig, (var_ax1, var_ax2, var_ax3) = plt.subplots(nrows=1, ncols=3,figsize=(10.0,4.8))
    var_title = "/figure.png"
    for j in range(len(var_likelihood_columns)):  # going through each body part
        var_ax2.plot(var_likelihood_binary[j],label="{}".format(var_likelihood_columns[j][1]))  # plot the line of the binary likelihood
        var_ax3.plot(list(var_likelihood[var_likelihood_columns[j]].index / len(var_likelihood[var_likelihood_columns[j]])),var_likelihood_binary[j],label="{}".format(var_likelihood_columns[j][1]))  # plot the line of the binary likelihood
        var_ax1.plot([0, 1], [0, 0], label="{}".format(var_likelihood_columns[j][1]))
        var_ax1.legend()  # making the legend
    var_fig.savefig(var_figure_save_path+var_title,dpi=1200)  # saving the picture at high quality

#func_plot_save(mice_cam5_likelihood_binary,mice_cam5_likelihood,mice_cam5_likelihood_columns,figure_save_path)
#func_plot_save(mice_cam6_likelihood_binary,mice_cam6_likelihood,mice_cam6_likelihood_columns,figure_save_path)
#func_plot_save(eyelid_left_likelihood_binary,eyelid_left_likelihood,eyelid_left_likelihood_columns,figure_save_path)
#func_plot_save(eyelid_right_likelihood_binary,eyelid_right_likelihood,eyelid_right_likelihood_columns,figure_save_path)





def func_smooth_sequences(var_compressed_sequences): #takes the compressed sequences for all body parts and return smooths sequences for all bodyparts
    var_smooth_sequences = [] #smooth sequences for all bodyparts
    for var_compressed_sequences_per_bodypart in var_compressed_sequences: #look at a specific bodypart
        var_smooth_sequences_per_bodypart = [] #make a list for specific bodypart
        for var_sequence in var_compressed_sequences_per_bodypart: #look at a sequence for one bodypart
            var_first_sequence_index = var_sequence[0] #look at the first index in de list
            var_last_sequence_index = var_sequence[len(var_sequence)-1] #look at the last index in the list
            var_smooth_single_sequence = list(range(var_first_sequence_index,var_last_sequence_index+1)) #make an integers list that goes from the first to the last index
            var_smooth_sequences_per_bodypart.append(var_smooth_single_sequence) #add this smooth sequence to the bodypart list
        var_smooth_sequences.append(var_smooth_sequences_per_bodypart) #add the bodypart list to the list for all bodyparts
    return var_smooth_sequences

mice_cam5_smooth_sequences = func_smooth_sequences(mice_cam5_compressed_sequences)
mice_cam6_smooth_sequences = func_smooth_sequences(mice_cam6_compressed_sequences)
eyelid_left_smooth_sequences = func_smooth_sequences(eyelid_left_compressed_sequences)
eyelid_right_smooth_sequences = func_smooth_sequences(eyelid_right_compressed_sequences)





def func_frames_to_time(var_npz_file,var_smooth_sequences):
    var_timestamps = var_npz_file["timestamps"]
    var_time_sequences = []
    for var_sequences_per_bodypart in var_smooth_sequences:
        var_time_sequences_per_bodypart = []
        for var_sequence in var_sequences_per_bodypart:
            var_time_single_sequence = []
            for var_index_value in var_sequence:
                if var_index_value < len(var_timestamps):
                    var_time_value = var_timestamps[var_index_value]
                    var_time_single_sequence.append(var_time_value)
            var_time_sequences_per_bodypart.append(var_time_single_sequence)
        var_time_sequences.append(var_time_sequences_per_bodypart)
    return var_time_sequences

mice_cam5_smooth_time_sequences = func_frames_to_time(mice_cam5_npz,mice_cam5_smooth_sequences)
mice_cam6_smooth_time_sequences = func_frames_to_time(mice_cam6_npz,mice_cam6_smooth_sequences)
eyelid_left_smooth_time_sequences = func_frames_to_time(eyelid_left_npz,eyelid_left_smooth_sequences)
eyelid_right_smooth_time_sequences = func_frames_to_time(eyelid_right_npz,eyelid_right_smooth_sequences)





def func_video_writer(var_video_path,var_compressed_sequences_per_bodypart): #input is the video path and the compressed sequences FOR ONE BODYPART!!!!! (so "eyelid_left_compressed_sequences[4]" for the closed eyelid)
    var_output_path = os.path.splitext(var_video_path)[0] + '_converted_video' + '.mp4' #choose the output path for the video
    var_reader = imageio.get_reader(var_video_path) #read the video

    var_test_frame = var_reader.get_data(0) #look at the data of the test frame
    var_black_frame = np.zeros([var_test_frame.shape[0], var_test_frame.shape[1], var_test_frame.shape[2]], dtype=np.uint8) #set the white/black frame to the same size as the test frame
    var_black_frame.fill(255)  # or img[:] = 255

    var_fps = var_reader.get_meta_data()['fps'] #look at the amount of frames per second of the original video
    var_writer = imageio.get_writer(var_output_path,fps=var_fps) #let the new video have the same amount of fps as the original video

    for var_sequence in var_compressed_sequences_per_bodypart: #look at a sequence
        for var_index in var_sequence: #look at one frame index
            var_frame = var_reader.get_data(var_index) #read this specific frame
            var_writer.append_data(var_frame) #write this frame to the video
        for i in range(int(var_fps/2)): #make sure that a 1/3 of one second a white/black frame is shown between different sequences.
            var_writer.append_data(var_black_frame) #so then you add the black frames

#func_video_writer(eyelid_left_video_path ,eyelid_left_smooth_sequences[4])
#func_video_writer(eyelid_right_video_path,eyelid_right_smooth_sequences[4])





def func_x_y_eyelid_plot(var_mice_timestamps, var_mice_x_y, var_mice_x_y_columns, var_fig_path,var_mice_cam5): #this function takes for one camera the timestamps, the x- or y-h5-file, column file of x,y , the path to save the figure and a string with "eyelid_left" for example as title.
    var_t = var_mice_timestamps #copying the timestamps as simpler variable
    var_x_y_interpolated = [] #get for x or y a list per bodypart where the interpolated values are in.
    for var_column_per_bodypart in var_mice_x_y_columns: #look at one bodypart
        var_x_y_per_bodypart = var_mice_x_y[var_column_per_bodypart].array.__array__()[:len(var_mice_timestamps)] #get the x data for one bodypart
        var_f = interp1d(var_t, var_x_y_per_bodypart) #make an interpolation function with the data
        var_x_y_interpolated_per_bodypart = var_f(var_t) #fill in the time and get interpolated x of y values
        var_x_y_interpolated.append(var_x_y_interpolated_per_bodypart) #add one bodypart to the big list

    var_fig, var_ax = plt.subplots(nrows=len(var_mice_x_y_columns), ncols=1, sharex=True) #make a figure with different plot with shared x values
    var_title = var_mice_cam5 + "_" + var_mice_x_y_columns[0][2] #make the title for the figure and saving path
    var_fig.suptitle(var_title) #give the title to the figure
    for var_i in range(len(var_ax)): #look at all the bodypart axes.
        var_ax[var_i].plot(var_t, var_x_y_interpolated[var_i]) #plot (t,x) where x (or y) is interpolated
        var_ax[var_i].set_title(var_mice_x_y_columns[var_i][1]) #set the title to the bodypart (nasal edge for example)
        var_ax[var_i].set_ylabel("time") #give every
    var_ax[len(var_ax)-1].set_xlabel(var_mice_x_y_columns[len(var_ax)-1][2])
    var_fig.savefig(var_fig_path + "/" + var_title, dpi=1200)

#func_x_y_eyelid_plot(eyelid_left_timestamps, eyelid_left_x, eyelid_left_x_columns,figure_save_path,"eyelid_left")
#func_x_y_eyelid_plot(eyelid_left_timestamps, eyelid_left_y, eyelid_left_y_columns,figure_save_path,"eyelid_left")
#func_x_y_eyelid_plot(eyelid_right_timestamps, eyelid_right_x, eyelid_right_x_columns,figure_save_path,"eyelid_right")
#func_x_y_eyelid_plot(eyelid_right_timestamps, eyelid_right_y, eyelid_right_y_columns,figure_save_path,"eyelid_right")

t2 = time.time()

print('The time it took for running is {} seconds'.format(t2-t1))


#the script works if I enter everything above here at once


