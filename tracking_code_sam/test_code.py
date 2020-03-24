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






def func_frame_of_lowest_likelihood(var_sequences):
    var_lowest_likelihood_index = [] #Making a list (within it a list) from the indices where a sequence has a minimal likelihood
    var_lowest_likelihood_values = [] #Making the list with corresponding likelihoods
    for j in var_sequences: #look at the 15 bodyparts
        var_lowest_likelihood_index_per_bodypart = [] #look at index per bodypart
        var_lowest_likelihood_values_per_bodypart = [] #look at the likelihood per bodypart
        for i in j: #look at the list-sequences per bodypart
            index_min = i.idxmin() #look at the index corresponding to the minimal likelihood in a sequence
            likelihood_min = min(i.array) #looking at the minimal likelihood of a sequence
            var_lowest_likelihood_index_per_bodypart.append(index_min) #add the index to the index_list per bodypart
            var_lowest_likelihood_values_per_bodypart.append(likelihood_min) #add the likelihood to the likelihood_list per bodypart
        var_lowest_likelihood_index.append(var_lowest_likelihood_index_per_bodypart) #add each index bodypart list to the big index list
        var_lowest_likelihood_values.append(var_lowest_likelihood_values_per_bodypart) #add each likelihood bodypart list ot the big likelihood bodypart list
    return(var_lowest_likelihood_index,var_lowest_likelihood_values)
lowest_likelihood_index, lowest_likelihood_values = func_frame_of_lowest_likelihood(sequences)

#Conclusion:
#       There are 2 lists "low_likelihood_reduced_frames_index_single_frame" and
#       "low_likelihood_reduced_frames_likelihood_single_frame". These lists
#       contain 15 elements each (for each body part one list). The lists per bodypart contain the lowest values
#       of likelihood per sequence (low likelihood sequence with following frames that have p<0.99).




def func_binary(var_likelihood,var_likelihood_columns,var_threshold):
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

likelihood_binary = func_binary(likelihood,likelihood_columns,threshold)




for j in range(len(likelihood_columns)): #going through each body part
    plt.plot(likelihood_binary[j], label="{}".format(likelihood_columns[j][1])) #plot the line of the binary likelihood
    plt.legend() #making the legend
plt.savefig('/Users/samsuidman/Desktop/likelihood_figures/plaatje.png',dpi=1200) #saving the picture at high quality



fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
for j in range(len(likelihood_columns)): #going through each body part
    ax2.plot(likelihood_binary[j], label="{}".format(likelihood_columns[j][1])) #plot the line of the binary likelihood
    ax3.plot(list(likelihood[likelihood_columns[j]].index / len(likelihood[likelihood_columns[j]])), likelihood_binary[j], label="{}".format(likelihood_columns[j][1]))  # plot the line of the binary likelihood
    ax1.plot([0,1],[0,0], label="{}".format(likelihood_columns[j][1]))
    ax1.legend() #making the legend
fig.savefig('/Users/samsuidman/Desktop/likelihood_figures/plaatje.png',dpi=1200) #saving the picture at high quality


def func_plot(var_likelihood_binary,var_likelihood_columns,var_figure_save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(10.0,4.8))
    for j in range(len(var_likelihood_columns)):  # going through each body part
        ax2.plot(var_likelihood_binary[j],label="{}".format(likelihood_columns[j][1]))  # plot the line of the binary likelihood
        ax3.plot(list(var_likelihood[var_likelihood_columns[j]].index / len(var_likelihood[var_likelihood_columns[j]])),var_likelihood_binary[j],label="{}".format(var_likelihood_columns[j][1]))  # plot the line of the binary likelihood
        ax1.plot([0, 1], [0, 0], label="{}".format(var_likelihood_columns[j][1]))
        ax1.legend()  # making the legend
    fig.savefig(var_figure_save_path,dpi=1200)  # saving the picture at high quality

func_plot(likelihood_binary,likelihood_columns,'/Users/samsuidman/Desktop/likelihood_figures/plaatje.png')




def func_npz_reader(var_path_to_npz_file):
    var_npz_file = np.load(var_path_to_npz_file)
    return var_npz_file




def func_h5_timestamps_length(var_h5_file,var_npz_file):
    for i in



eyelid_left_npz.files





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


