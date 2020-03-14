import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os.path
import time

t1 = time.time()





#reading the tracking files (h5), these variables now contain a matrix, 3 elements per
#bodypart (x,y,likelihood). There are 10 bodyparts, so in total there are 30 elements.
#Each element got almost 20000 values, which are coordinate-values (for x and y) or
#likelihoodvalues
def func_working_path(var_working_path): #returns an absolute working path as a variable
    var_abs_working_path = os.path.abspath(var_working_path)
    return var_abs_working_path

mice_h5_path = func_working_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5")
mice_video_path = func_working_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000_labeled.mp4")





def func_h5_reader(var_path_to_h5_file): #reads a h5 file using the path to the file as a variable
    var_h5_file = pd.read_hdf(var_path_to_h5_file)
    return var_h5_file

mice_h5 = func_h5_reader(mice_h5_path)





def func_likelihood_columns(var_mice_h5): #takes a h5_file with (x,y,likelihood)-data (including column names) and returns a list of the likelihood column names
    var_likelihood_columns = [] #make a list of likelihood column names (this is a vector)
    for i in var_mice_h5.columns:
        if i[2] == 'likelihood':
            var_likelihood_columns.append(i)
    return var_likelihood_columns

likelihood_columns = func_likelihood_columns(mice_h5)





def func_likelihood(var_likelihood_columns,var_mice_h5): #takes a list h5 file with (x,y,likelihood)-data and a list of likelihood-column-names and returns a matrix of only the likelihood-data labeled by the column names
    var_likelihood = var_mice_h5[var_likelihood_columns] # make a list of likelihoodvalues per tracking object (so this is a matrix)
    return var_likelihood

likelihood = func_likelihood(likelihood_columns,mice_h5)





def func_video(var_video_path): #takes a path to a video and returns the metadata and the amount of frames of the video (as tuple)
    with imageio.get_reader(var_video_path) as var_video:     #with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
        var_video_meta_data = var_video.get_meta_data() #this contains the metadata, such as  fps (frames per second), duration, etc.
        var_video_frames = var_video.count_frames() #counting the amount of frames (=19498)
    #    var_video_dataframe_2364 = var_video.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 19497
    return var_video_meta_data, var_video_frames

video_info = func_video(mice_video_path)





def func_low_likelihood(var_likelihood,var_likelihood_columns,threshold): #takes likelihood-matrix (it is matrix, because of multiple bodyparts), a likelihood-column-name array and a threshold and returns a matrix of low likelihoods and a matrix of low likelihood indices
    var_low_likelihood_values = []
    var_low_likelihood_index = []
    for i in var_likelihood_columns:
        var_low_likelihood_values_per_bodypart = var_likelihood[i].array[var_likelihood[i].array<threshold]
        var_low_likelihood_index_per_bodypart = var_likelihood[i].index[var_likelihood[i].array<threshold]
        var_low_likelihood_values.append(var_low_likelihood_values_per_bodypart)
        var_low_likelihood_index.append(var_low_likelihood_index_per_bodypart)
    return(var_low_likelihood_values,var_low_likelihood_index)

low_likelihood_values,low_likelihood_index = func_low_likelihood(likelihood,likelihood_columns,0.99)





#To plot bodypart i (0-14) you can do:
#plt.plot(low_likelihood_index[i])
#title = "{}".format(likelihood_columns[i][1])
#plt.title(title)
#plt.show()





def func_low_likelihood_sequences(var_low_likelihood_values,var_low_likelihood_index,var_likelihood_columns): #takes a list of low-likelihood-values, low-likelihood-indices and likelihood-column-names. Returns a list (15 elements for 15 bodyparts), where for each element there is a list with sequences of low-likelihood-frames. These for each sequences you can call (sequencelist.index) for an array of indices or (sequencelist.array) for an array of likelihoods.

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
            subsequent_likelihood = likelihood[likelihood_columns[j]][k]  # make a small list of likelihoods that match the small list of indices
            var_sequences_values_per_bodypart.append(subsequent_likelihood)  # add this small likelihood list to the list of one body part
        var_sequences.append(var_sequences_values_per_bodypart)  # add the one body part list to the list of all bodyparts

    return var_sequences

sequences = func_low_likelihood_sequences(low_likelihood_values,low_likelihood_index,likelihood_columns)





low_likelihood_reduced_frames_index_single_frame = [] #Making a list (within it a list) from the indices where a sequence has a minimal likelihood
low_likelihood_reduced_frames_likelihood_single_frame = [] #Making the list with corresponding likelihoods
for j in low_likelihood_reduced_frames_likelihood: #look at the 15 bodyparts
    low_likelihood_reduced_frames_index_single_frame_column = [] #look at index per bodypart
    low_likelihood_reduced_frames_likelihood_single_frame_column = [] #look at the likelihood per bodypart
    for i in j: #look at the list-sequences per bodypart
        index_min = i.idxmin() #look at the index corresponding to the minimal likelihood in a sequence
        likelihood_min = min(i.array) #looking at the minimal likelihood of a sequence
        low_likelihood_reduced_frames_index_single_frame_column.append(index_min) #add the index to the index_list per bodypart
        low_likelihood_reduced_frames_likelihood_single_frame_column.append(likelihood_min) #add the likelihood to the likelihood_list per bodypart
    low_likelihood_reduced_frames_index_single_frame.append(
        low_likelihood_reduced_frames_index_single_frame_column) #add each index bodypart list to the big index list
    low_likelihood_reduced_frames_likelihood_single_frame.append(
        low_likelihood_reduced_frames_likelihood_single_frame_column) #add each likelihood bodypart list ot the big likelihood bodypart list
#Conclusion:
#       There are 2 lists "low_likelihood_reduced_frames_index_single_frame" and
#       "low_likelihood_reduced_frames_likelihood_single_frame". These lists
#       contain 15 elements each (for each body part one list). The lists per bodypart contain the lowest values
#       of likelihood per sequence (low likelihood sequence with following frames that have p<0.99).





likelihood_binary = [] #Making a list for all bodyparts (len(list)=15) where if p<0.99 the value of list[i] becomes NaN and otherwise becomes 1+i (so that the lines are above eachother)
k=0 #place for the first line
for i in likelihood_columns: #going through each bodypart
    likelihood_binary_column = [] #making a list per bodypart where [1,NaN,NaN,1,1,...]
    for j in likelihood[i]: #cloning the likelihood list of a bodypart [0.3338,0.9783,...]
        likelihood_binary_column.append(j) # " " "
    likelihood_binary_column = np.asarray(likelihood_binary_column) #making type=array of it
    likelihood_binary_column[likelihood_binary_column<likelihood_threshold] = np.NaN #setting p<0.99 to not a number
    likelihood_binary_column[likelihood_binary_column>likelihood_threshold] = len(likelihood_columns)-k #setting p>0.99 to 15 (or 14,13,...,2,1) so that all lines can be plotted above eachother)
    likelihood_binary.append(likelihood_binary_column) #adding the list of one bodypart to the bigger list
    k+=1 #setting k+1 so that the next bodypart comes at the line above/below the other

for j in range(len(likelihood_columns)): #going through each body part
    plt.plot(likelihood_binary[j], label="{}".format(likelihood_columns[j][1])) #plot the line of the binary likelihood
    plt.legend() #making the legend
plt.savefig('/Users/samsuidman/Desktop/likelihood_figures/plaatje.png',dpi=1200) #saving the picture at high quality







t2 = time.time()

print('The time it took for running is {}'.format(t2-t1))


#the script works if I enter everything above here at once (and takes about 63 seconds to run)


