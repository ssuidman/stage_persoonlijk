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