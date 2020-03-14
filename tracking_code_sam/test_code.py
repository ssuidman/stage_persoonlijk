#import the package with which you can save paths easily
import os.path
#To save a path use abspath, this can be the full path, but also a part of the path
billie_mp4_path = os.path.abspath("/Users/samsuidman/Downloads/no_time_to_die.mp4")

#import the package for video/image operations
import imageio
#with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
with imageio.get_reader("/Users/samsuidman/Downloads/no_time_to_die.mp4") as billie: #instead writing the path you can also write --> billie_mp4_path
    metadata = billie.get_meta_data() #this contains the metadata, such as  fps (frames per second), duration, etc.
    frames = billie.count_frames() #counting the amount of frames (=6365)
    data = billie.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 6364
metadata
frames
data

#import plt
import matplotlib.pyplot as plt
plt.imshow(data)
plt.show()

fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
for j in range(len(likelihood_columns)): #going through each body part
    ax2.plot(likelihood_binary[j], label="{}".format(likelihood_columns[j][1])) #plot the line of the binary likelihood
    ax3.plot(list(likelihood[likelihood_columns[j]].index / len(likelihood[likelihood_columns[j]])), likelihood_binary[j], label="{}".format(likelihood_columns[j][1]))  # plot the line of the binary likelihood
    ax1.plot([0,1],[0,0], label="{}".format(likelihood_columns[j][1]))
    ax1.legend() #making the legend
fig.savefig('/Users/samsuidman/Desktop/likelihood_figures/plaatje.png',dpi=1200) #saving the picture at high quality


import os.path
def func_path_names(var_path):
    var_abspath = os.path.abspath(var_path)
    return var_abspath

mice_h5_path = func_path_names('/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5')
mice_h5 = pd.read_hdf(mice_h5_path)


import os
def func_working_path(var_working_path):
    var_abs_working_path = os.path.abspath(var_working_path)
    return var_abs_working_path

oefenpad = func_working_path('/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure')



def func_video(var_video_path): #takes a path to a video and returns the metadata and the amount of frames of the video (as tuple)
    with imageio.get_reader(var_video_path) as var_video:     #with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
        var_video_meta_data = var_video.get_meta_data() #this contains the metadata, such as  fps (frames per second), duration, etc.
        var_video_frames = var_video.count_frames() #counting the amount of frames (=19498)
    #    var_video_dataframe_2364 = var_video.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 19497
    return var_video_meta_data, var_video_frames
video_info = func_video(mice_video_path)



def func_low_likelihood(var_likelihood,var_likelihood_columns,threshold):
    var_low_likelihood_values = []
    var_low_likelihood_index = []
    for i in var_likelihood_columns:
        var_low_likelihood_values_per_bodypart = var_likelihood[i].array[var_likelihood[i].array<threshold]
        var_low_likelihood_index_per_bodypart = var_likelihood[i].index[var_likelihood[i].array<threshold]
        var_low_likelihood_values.append(var_low_likelihood_values_per_bodypart)
        var_low_likelihood_index.append(var_low_likelihood_index_per_bodypart)
    return(var_low_likelihood_values,var_low_likelihood_index)
low_likelihood_values,low_likelihood_index = func_low_likelihood(likelihood,likelihood_columns,0.99)



    low_likelihood_reduced_frames_index = [] #making a list for all body parts, where the indices of frames with low likelihood are in, but reduced so that subsequent frames are not in it
    for i in range(len(likelihood_columns)): #go through all body parts
        low_likelihood_reduced_frames_index_column = [] #making a list for one body part. Several lists of indices of subsequent low likelihoods are stored in this bigger list
        for first_low_likelihood_frame in low_likelihood_all_frames[i]: #go through the list of all low likelihood frames
            if first_low_likelihood_frame-1 not in low_likelihood_all_frames[i]: #this is so that no lists are made that contain the same elements as another list
                second_variable = first_low_likelihood_frame #set a second variable to avoid conflict with the first one
                subsequent_list = [] #making a list where subsequent low likelihood indices can be stored in
                subsequent_list.append(second_variable) #append the first low likelihood value to this list
                while second_variable+1 in low_likelihood_all_frames[i]: #if the subsequent index of the low likelihood list is also in the low likelihood list, then look at this next index
                    second_variable+=1 #look at the next index
                    subsequent_list.append(second_variable) #add the next index also to the list
                low_likelihood_reduced_frames_index_column.append(subsequent_list) #add the list with subsequent indices to the list for one bodypart.
        low_likelihood_reduced_frames_index.append(low_likelihood_reduced_frames_index_column) #add the list for one bodypart to the list for all bodyparts
    #Now there is a list "low_likelihood_reduced_frames_index". For example "low_likelihood_reduced_frames_index[3][5:8]" gives now for the third bodypart: the fourth till the seventh subsequent low likelihood frames in lists .



def func_low_likelihood_sequences(var_low_likelihood_values,var_low_likelihood_index,var_likelihood_columns):
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
    return var_sequences_index
sequences_index = func_low_likelihood_sequences(low_likelihood_values,low_likelihood_index,likelihood_columns)



var_sequences_values = [] #list where all reduced likelihoods for all bodyparts come in
for j in range(len(var_likelihood_columns)): #getting across all bodypart-indices (0-14)
    var_sequences_values_per_bodypart = [] #likelihood list per bodypart
    for k in var_sequences_index[j]: #look at a specific list with subsequent indices in a bodypart list
        subsequent_likelihood = likelihood[likelihood_columns[j]][k] #make a small list of likelihoods that match the small list of indices
        var_sequences_values_per_bodypart.append(subsequent_likelihood) #add this small likelihood list to the list of one body part
    var_sequences_values.append(var_sequences_values_per_bodypart) #add the one body part list to the list of all bodyparts

def func_low_likelihood_sequences(var_low_likelihood_values,var_low_likelihood_index,var_likelihood_columns):

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

