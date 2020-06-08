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





def func_compressed_sequences(var_high_sequences,var_continued_frames): #takes 2 things: 1) a big list with 15 elements (each bodypart), each
    # with a list of sequences (sequences are pandas.series.Series) 2) the maximum amount of frames (plus 1) that can be between to sequences
    # when merging sequences together. The function returns a big list containing lists with sequences that are merged together.
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
                var_high_sequences_compressed_per_bodypart_temp.extend(var_high_sequences_per_bodypart[j].index) #you merge the sequences that should be together            var_high_sequences_compressed_per_bodypart.append(var_high_sequences_compressed_per_bodypart_temp) #you add this list of compressed sequences to the big list
        var_high_sequences_compressed.append(var_high_sequences_compressed_per_bodypart)
    return(var_high_sequences_compressed) #you give back a big list that contains lists of compressed sequences

eyelid_left_compressed_sequences = func_compressed_sequences(eyelid_left_high_sequences,200)
eyelid_right_closed_eye_sequences = func_compressed_sequences(eyelid_right_high_sequences,200)


reader = imageio.get_reader('/Users/samsuidman/Desktop/video_test_map/rpi_camera_4.mp4')
writer = imageio.get_writer('/Users/samsuidman/Desktop/video_test_map/rpi_camera_4_writer.mp4',fps=2)
frame = []
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][0]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][1]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][2]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][3]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][4]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][5]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][6]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][7]))
frame.append(reader.get_data(eyelid_left_compressed_sequences[4][1][8]))
writer.append_data(frame[0])
writer.append_data(frame[1])
writer.append_data(frame[2])
writer.append_data(frame[3])
writer.append_data(frame[4])
writer.append_data(frame[5])
writer.append_data(frame[6])
writer.append_data(frame[7])
writer.append_data(frame[8])
writer.close()



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
func_video_writer('/Users/samsuidman/Desktop/video_test_map/rpi_camera_4.mp4','/Users/samsuidman/Downloads/zwart_foto.jpg',eyelid_right_compressed_sequences[4])






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




def func_indices_to_timestamps(var_index_sequences):
    var_timestamps_sequences = []
    for var_index_sequences_per_bodypart in var_index_sequences:
        for var_index_single_sequence in var_index_sequences_per_bodypart:




def func_x_y_eyelid_plot(var_mice_timestamps, var_mice_x_y, var_mice_x_y_columns, var_fig_path,var_mice_cam5):
    var_t = var_mice_timestamps
    var_x_y_interpolated = []
    for var_column_per_bodypart in var_mice_x_y_columns:
        var_x_y_per_bodypart = var_mice_x_y[var_column_per_bodypart].array.__array__()[:len(var_mice_timestamps)]
        var_f = interp1d(var_t, var_x_y_per_bodypart)
        var_x_y_interpolated_per_bodypart = var_f(var_t)
        var_x_y_interpolated.append(var_x_y_interpolated_per_bodypart)

    var_fig, var_ax = plt.subplots(nrows=len(var_mice_x_y_columns), ncols=1, sharex=True)
    var_title = var_mice_cam5 + "_" + var_mice_x_y_columns[0][2]
    var_fig.suptitle(var_title)
    for var_i in range(len(var_ax)):
        var_ax[var_i].plot(var_t, var_x_y_interpolated[var_i])
        var_ax[var_i].set_title(var_mice_x_y_columns[var_i][1])
        var_ax[var_i].set_ylabel("time")
    var_ax[len(var_ax)-1].set_xlabel(var_mice_x_y_columns[len(var_ax)-1][2])
    var_fig.savefig(var_fig_path + "/" + var_title, dpi=1200)

func_x_y_eyelid_plot(eyelid_left_timestamps, eyelid_left_x, eyelid_left_x_columns,figure_save_path,"eyelid_left")








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

data = func_four_images(data0_right,data0_left,data0_cam5,data0_cam6)
writer = imageio.get_writer("/Users/samsuidman/Desktop/video_test_map/four_images.png")
writer.append_data(data)



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




@click.command(name='distance-speed-plot')
@click.argument('db_path', type=click.Path())
@click.option('--mouse', '-m', default=['M3728', 'M3729', 'M4081'], multiple=True)

#this function makes plots of the distance from each m2 body part to each eyecam of m1
def cli_distance_speed_plot(db_path,
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

        fig, ax = plt.subplots(nrows=2, ncols=len(part_names_m2), sharex=True, sharey=False, figsize=[32.4, 8.8])
        for i in range(len(eyes)):
            for j in range(len(part_names_m2)):
                ax[i][j].scatter(np.concatenate(eye_data1[eyes[i]]['closed_eye_distance'][part_names_m2[j]]),
                                 np.concatenate(eye_data1[eyes[i]]['closed_eye_speed'][part_names_m2[j]]))
                ax[i][j].set_title('%s' % part_names_m2[j].replace('_', ' '))
                ax[i][j].set_xlabel('y (cm)')
                ax[i][j].set_ylabel('y (cm/s)')
        fig.tight_layout()
        fig.show()

    return tracking_data, eye_data

cli.add_command(cli_distance_speed_plot)


db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M4081']
tracking_data1,eye_data1 = cli_distance_speed_plot(db_path1,mouse1)






#This is a plot of all body_parts where the averaged speed is plotted (of the tracking_data)
#fig,ax = plt.subplots(ncols=len(tracking_data1['body_parts'].keys()),figsize=[32.4,4.8])
#for i in range(len(tracking_data1['body_parts'].keys())):
#    ax[i].plot(tracking_data1['body_parts'][list(tracking_data1['body_parts'].keys())[i]]['speed_averaged'])
#    ax[i].set_title(list(tracking_data1['body_parts'].keys())[i])
#fig.tight_layout()
#fig.show()


# cam3, cam4 --> fps=60
# cam5, cam6 --> fps=30

part_names_m2 # is a list of body part names
eyes #is a list of eye names (so left and right)
ax[1][3] #is the second row, fourth coloumn
fig,ax = plt.subplots(nrows=2, ncols=len(part_names_m2), sharex=True, sharey=False,figsize=[32.4,8.8])
for i in range(len(eyes)):
    for j in range(len(part_names_m2)):
        ax[i][j].scatter(np.concatenate(eye_data1[eyes[i]]['closed_eye_distance'][part_names_m2[j]]),np.concatenate(eye_data1[eyes[i]]['closed_eye_speed'][part_names_m2[j]]))
        ax[i][j].set_title('%s' % part_names_m2[j].replace('_', ' '))
        ax[i][j].set_xlabel('y (cm)')
        ax[i][j].set_ylabel('y (cm/s)')
fig.tight_layout()
fig.show()




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

        fig, ax = plt.subplots(nrows=2, ncols=len(part_names_m2), sharex=False, sharey=False, figsize=[20, 5])
        for i in range(len(eyes)):
            for j in range(len(part_names_m2)):
                ax[i][j].hist(np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][part_names_m2[j]]), bins=12,
                              range=(0, 30))
                ax[i][j].set_title('%s' % part_names_m2[j].replace('_', ' '))
                ax[i][j].set_xlabel('distance (cm)')
                ax[i][j].set_ylabel('#counts')
                ax[i][j].set_xticks([0, 5, 10, 15, 20, 25, 30])
        fig.tight_layout()
    plt.show(block=True)







@click.command(name='minimal-distance-bodyparts')
@click.argument('db_path', type=click.Path())
@click.option('--mouse', '-m', default=['M3728', 'M3729', 'M4081'], multiple=True)


def cli_minimal_distance_bodyparts(db_path,mouse=None):
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

        eyes = list(eye_data.keys())
        m2_bodyparts = list(eye_data['left']['closed_eye_distance'].keys())

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=[15, 5])
        for i in range(len(eyes)):
            for j in range(2):
                if j == 0: #minimal distance
                    distance_bodypart_matrix = np.array([np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][m2_bodypart]) for m2_bodypart in m2_bodyparts])
                    min_distance = np.nanmin(distance_bodypart_matrix, axis=0)
                    ax[i][j].hist(min_distance, bins=16,range=(0, 30))
                    ax[i][j].set_title('%s %s' % (eyes[i],'minimal distance'))
                    ax[i][j].set_xlabel('distance (cm)')
                    ax[i][j].set_ylabel('#counts')
                    ax[i][j].set_xticks([5, 15, 25])
                if j==1: #closest bodypart
                    distance_bodypart_matrix = np.array([np.concatenate(eye_data[eyes[i]]['closed_eye_distance'][m2_bodypart]) for m2_bodypart in m2_bodyparts])
                    closest_bodypart = [i for c in range(len(distance_bodypart_matrix[0])) for i,j in enumerate(distance_bodypart_matrix[:,c]) if j==np.nanmin(distance_bodypart_matrix[:,c])]
                    ax[i][j].hist(closest_bodypart, bins=8, range=(0,8),align='left')
                    ax[i][j].set_title('%s %s' % (eyes[i], 'closest bodypart'))
                    ax[i][j].set_ylabel('#counts')
                    ax[i][j].set_xticklabels(labels=m2_bodyparts, rotation=45, rotation_mode="anchor")
        fig.tight_layout()
    plt.show(block=True)

cli.add_command(cli_minimal_distance_bodyparts)


db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M4081']
cli_distance_bodyparts(db_path1,mouse1)


#here the amount of counts and the percentage can be calculated.

def cli_closed_eye_nan(eye_data):
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





@click.command(name='closed-eye-nan-counts')
@click.argument('db_path', type=click.Path())
@click.option('--mouse', '-m', default=['M3728', 'M3729', 'M4081'], multiple=True)

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

        part_names_m2 = sorted([k for k in tracking_data['body_parts'].keys() if k.startswith('m2')])
        eyes = list(eye_data.keys())

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
        print(nan_dict)

cli.add_command(cli_closed_eye_nan)



eye_closed_interval = [np.asarray(range(c[0],c[1]+1)) for c in eye_data['left']['eye_closed_interval']]
eye_closed_interval2 = np.concatenate(eye_closed_interval)


eye_closed_timestamps = np.concatenate([eye_data['left']['timestamps'][c[0]:c[1]+1] for c in eye_data['left']['eye_closed_interval']])
all_timestamps = eye_data['left']['timestamps']
timestamps_with_zeros = [c if c in eye_closed_timestamps else 0 for c in all_timestamps]
interp_f_zeros = interpolate.interp1d(all_timestamps,timestamps_with_zeros)
timestamps_zeros_tracking_data = interp_f_zeros(tracking_data['timestamps'])
indices_tracking_data =






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




import imageio
#with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
with imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_15-06-35_Silence_box_no_enclosure_M3729/rpi_camera_4.mp4") as billie: #instead writing the path you can also write --> billie_mp4_path
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

##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################



# What do I need?
# --> video frames indices of cam5 of cam6 (maybe from timestamps) where eye closure takes place.

#First import func_video_writer from DLC_Mice_script_functions
#This makes a video of cam5 for the closed eye intervals:
#func_video_writer("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam5/raw_video/rpi_camera_5.mp4",[eye_closed_tracking_data_interval])

# --> (high likelihood) x,y coordinates of m1 eyes/nose from the eye_closure indices
def f(eye_data,tracking_data,h5_file,db_path):

    #First add to the tracking_data, the frames where eyes are closed
    tracking_data['eye_closed_interval'] = {}
    for eye in ['left','right']:
        eye_closed_timestamps_eye_data = np.concatenate([eye_data[eye]['timestamps'][c[0]:c[1] + 1] for c in eye_data[eye]['eye_closed_interval']])
        interp_f = interpolate.interp1d(tracking_data['timestamps'], list(range(len(tracking_data['timestamps']))))
        eye_closed_interval_tracking_data = (np.round(interp_f(eye_closed_timestamps_eye_data))).astype(int)
        tracking_data['eye_closed_interval'][eye] = eye_closed_interval_tracking_data

    #Then make a lists of left eye, right eye, nose tip, (x,y,likelihood) list where the likelihood>0.99 for all parts.
    eye_left = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_left']))[:len(tracking_data['timestamps'])]
    eye_right = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_right']))[:len(tracking_data['timestamps'])]
    nose_tip = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_nose_tip']))[:len(tracking_data['timestamps'])]
    eye_mid = np.array([[(eye_left[c, 0] + eye_right[c, 0]) / 2, (eye_left[c, 1] + eye_right[c, 1]) / 2] for c in range(len(tracking_data['timestamps']))])
    head_mid = np.array([[(eye_mid[c, 0] + nose_tip[c, 0]) / 2, (eye_mid[c, 1] + nose_tip[c, 1]) / 2] for c in range(len(tracking_data['timestamps']))])
    indices_99 = np.array([i for i in range(len(eye_left[:,2])) if eye_left[:,2][i]>0.99 and eye_right[:,2][i]>0.99 and nose_tip[:,2][i]>0.99])
#    eye_left_99 = eye_left[indices_99] #and for others

#make the
    reader = imageio.get_reader()
    for i in [indices_99[50], indices_99[100], indices_99[150], indices_99[200], indices_99[250]]:
        data = reader.get_data(i)
        plt.imshow(data)
        plt.scatter(eye_left[i, 0], eye_left[i, 1])
        plt.scatter(eye_right[i, 0], eye_right[i, 1])
        plt.scatter(nose_tip[i, 0], nose_tip[i, 1])
        plt.scatter(head_mid[i, 0], head_mid[i, 1])
        plt.show()
    reader.close()
#    transformed_positions,centers,angles = transformed_positions(tracking_data)
#    eye_left_transformed = np.array([[transformed_positions['m1_eyecam_left'][c,0],transformed_positions['m1_eyecam_left'][c,1],tracking_data['body_parts']['m1_eyecam_left']['likelihood'][c]] for c in range(len(tracking_data['body_parts']['m1_eyecam_left']['likelihood']))])
#    eye_right_transformed = np.array([[transformed_positions['m1_eyecam_right'][c, 0],transformed_positions['m1_eyecam_right'][c, 1],tracking_data['body_parts']['m1_eyecam_right']['likelihood'][c]] for c in range(len(tracking_data['body_parts']['m1_eyecam_right']['likelihood']))])
#    nose_tip_transformed = np.array([[transformed_positions['m1_nose_tip'][c, 0],transformed_positions['m1_nose_tip'][c, 1],tracking_data['body_parts']['m1_nose_tip']['likelihood'][c]] for c in range(len(tracking_data['body_parts']['m1_nose_tip']['likelihood']))])

    return eye_data,tracking_data

# --> video frame indices match h5_file indices

######For the sake of looking at good circle pictures, there is no need that this is a closed eye event for now
# --> so then pictures of closed eye events can be obtained with 3 x,y scatter points (for start begin with one closure index)
# --> With this information make a triangle of left eye, right eye, nose tip and set a circle point in the middle of these points (centroid, zwaartepunt)
# --> draw a circle of 30 mm around this center

def func_path(var_working_path): #returns an absolute working path as a variable
    var_abs_working_path = os.path.abspath(var_working_path)
    return var_abs_working_path
mice_cam5_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam5/mice/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5")
mice_cam6_h5_path = func_path("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/h5/M3728/together/cam6/mice/rpi_camera_6DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5")
def func_h5_reader(var_path_to_h5_file): #reads a h5 file using the path to the file as a variable
    var_h5_file = pd.read_hdf(var_path_to_h5_file)
    return var_h5_file
mice_cam5_h5 = func_h5_reader(mice_cam5_h5_path)
mice_cam6_h5 = func_h5_reader(mice_cam6_h5_path)


reader = imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/h5_video_results/video/M3728/together/cam5/raw_video/rpi_camera_5.mp4")
for i in [indices_99[50],indices_99[100],indices_99[150],indices_99[200],indices_99[250]]:
    data = reader.get_data(i)
    plt.imshow(data)
    plt.scatter(eye_left[i,0],eye_left[i,1])
    plt.scatter(eye_right[i,0],eye_right[i,1])
    plt.scatter(nose_tip[i,0],nose_tip[i,1])
    plt.scatter(head_mid[i,0],head_mid[i,1])
    plt.show()




# ------------------------------------------------------------------
# analysis 6: make a video of the two mice with their whisker range
# ------------------------------------------------------------------


def closed_eye_tracking_data(tracking_data,eye_data):
    #First add to the tracking_data, the frames where eyes are closed
    tracking_data['eye_closed_interval'] = {}
    for eye in ['left','right']:
        eye_closed_timestamps_eye_data = np.concatenate([eye_data[eye]['timestamps'][c[0]:c[1] + 1] for c in eye_data[eye]['eye_closed_interval']])
        interp_f = interpolate.interp1d(tracking_data['timestamps'], list(range(len(tracking_data['timestamps']))),fill_value=extrapolate)
        eye_closed_interval_tracking_data = (np.round(interp_f(eye_closed_timestamps_eye_data))).astype(int)
        tracking_data['eye_closed_interval'][eye] = eye_closed_interval_tracking_data
    return tracking_data


@click.command(name='whisker-video')
@click.argument('db_path', type=click.Path())
@click.argument('output_path', type=click.Path())
@click.option('--mouse', '-m', default=['M3728', 'M3729', 'M4081'], multiple=True)
@click.option('--camera_number', '-cam', default=6)
@click.option('--whisker_radius', '-r', default=1.5)

def cli_whisker_video(db_path, output_path, mouse=None, camera_number=6, whisker_radius=1.5):  # circle radius in cm
    assert camera_number in [5, 6]

    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

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

        whisker_circle = plt.Circle((mid_estimate_x[0], mid_estimate_y[0]), r, color='tab:red', fill=False, linewidth=1,label='whisker range')
        ax.add_patch(whisker_circle)  # adding whisker circle

        eye_left_circle = plt.Circle((eye_left[0, 0], eye_left[0, 1]), r/3, color='tab:green', label='eye')
        ax.add_patch(eye_left_circle)

        eye_right_circle = plt.Circle((eye_right[0, 0], eye_right[0, 1]), r/3, color='tab:green')
        ax.add_patch(eye_right_circle)

        head_mid_circle = plt.Circle((mid_estimate_x[0], mid_estimate_y[0]), r/3, color='tab:blue',label='head center')
        ax.add_patch(head_mid_circle)

        eyes_line = ax.plot((eye_left[0, 0], eye_right[0, 0]), (eye_left[0, 1], eye_right[0, 1]), '-', color='tab:orange', lw=2)[0]

        head_line = ax.plot((eye_mid[0, 0], mid_estimate_x[0]), (eye_mid[0, 1], mid_estimate_y[0]), '-', color='tab:orange', lw=2)[0]

        body_parts_m2 = [body_part for body_part in tracking_data['body_parts'].keys() if 'm2' in body_part.split('_')] #list of m2 body parts
        circles_m2 = {}
        for body_part_m2 in body_parts_m2:
            circle = plt.Circle((tracking_data['body_parts'][body_part_m2]['position'][0,0]*pix_per_cm,tracking_data['body_parts'][body_part_m2]['position'][1,0]*pix_per_cm))
            circles_m2[body_part_m2] = circle
            ax.add_patch(circle)

        fig.legend()

        percentage = 0  # starting percentag for iteration
        video_range = 500  # the total amount of video frames
        steps_of_showing = 100  # each hundredth (or what you want) is shown how much the video is processed
        hundredth = video_range / steps_of_showing

        with writer.saving(fig, op.abspath(output_path)+'/whisker_video_'+ mouse_id+'.mp4', dpi=dpi):
            for i in range(video_range):

                if i > hundredth:
                    percentage += 1
                    print(str(int(percentage * 100 / steps_of_showing)) + '%')
                    hundredth += video_range / steps_of_showing

                data = reader.get_data(i)
                img.set_data(data)
                eye_left_circle.set_center((eye_left[i, 0], eye_left[i, 1]))
                eye_right_circle.set_center((eye_right[i, 0], eye_right[i, 1]))
                whisker_circle.set_center((mid_estimate_x[i], mid_estimate_y[i]))
                head_mid_circle.set_center((mid_estimate_x[i], mid_estimate_y[i]))
                eyes_line.set_data((eye_left[i, 0], eye_right[i, 0]), (eye_left[i, 1], eye_right[i, 1]))
                head_line.set_data((eye_mid[i, 0], mid_estimate_x[i]), (eye_mid[i, 1], mid_estimate_y[i]))

                for body_part_m2 in body_parts_m2:
                    circles_m2[body_part_m2].set_center((tracking_data['body_parts'][body_part_m2]['position'][0,i]*pix_per_cm,tracking_data['body_parts'][body_part_m2]['position'][1,i]*pix_per_cm))

                writer.grab_frame()
        reader.close()
    return tracking_data, eye_data

cli.add_command(cli_whisker_video)

db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M4081']
tracking_data, eye_data = cli_whisker_video(db_path=db_path1, output_path="/Users/samsuidman/Desktop/video_test_map" , mouse=mouse1, camera_number=6, whisker_radius=1.5)

###############################




def cli_whisker_analysis(db_path,mouse=None,camera_number=6,whisker_radius=1.5):
    assert camera_number in [5, 6]

    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

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
        #eye_left = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_left']))[:len(tracking_data['timestamps'])]
        #eye_right = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_eyecam_right']))[:len(tracking_data['timestamps'])]
        #nose_tip = np.transpose(np.array([h5_file[c] for c in h5_file.keys() if c[1] == 'm1_nose_tip']))[:len(tracking_data['timestamps'])]

        eye_left = np.transpose(np.array([tracking_data['body_parts']['m1_eyecam_left']['position'][:,0],tracking_data['body_parts']['m1_eyecam_left']['position'][:,1],tracking_data['body_parts']['m1_eyecam_left']['likelihood']]))
        eye_right = np.transpose(np.array([tracking_data['body_parts']['m1_eyecam_right']['position'][:, 0],tracking_data['body_parts']['m1_eyecam_right']['position'][:, 1],tracking_data['body_parts']['m1_eyecam_right']['likelihood']]))
        nose_tip = np.transpose(np.array([tracking_data['body_parts']['m1_nose_tip']['position'][:, 0],tracking_data['body_parts']['m1_nose_tip']['position'][:, 1],tracking_data['body_parts']['m1_nose_tip']['likelihood']]))
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
        # look at the cm reference frame and where (x,y) for head center is and what the radius is in pixels
        mid_estimate_x_cm = mid_estimate_x / pix_per_cm
        mid_estimate_y_cm = mid_estimate_y / pix_per_cm
        r = whisker_radius * pix_per_cm

        body_parts_m2 = [body_part for body_part in tracking_data['body_parts'].keys() if 'm2' in body_part.split('_')] #list of m2 body parts
        fig,ax = plt.subplots(ncols=len(body_parts_m2),figsize=[15,5])
        for i,body_part_m2 in enumerate(body_parts_m2):
            body_part_x_cm = tracking_data['body_parts'][body_part_m2]['position'][:,0] #list of x values of bodypart
            body_part_y_cm = tracking_data['body_parts'][body_part_m2]['position'][:, 1] #list of x values of bodypart
            distance_cm = np.sqrt( (body_part_x_cm-mid_estimate_x_cm)**2 + (body_part_y_cm-mid_estimate_y_cm)**2 ) #distance in cm from head center
            tracking_data['body_parts'][body_part_m2]['inside_whisker_range'] = {}
            for eye in ['left', 'right']:
                distance_cm_eye_closed = np.array(
                    [j if i in tracking_data['eye_closed_interval'][eye] else np.nan for i, j in
                     enumerate(distance_cm)])  # filtered out the intervals with closed eyes
                distance_cm_effective_list = distance_cm_eye_closed[
                    np.isnan(distance_cm_eye_closed) == False]  # get out all the nan
                true_values = len([distance for distance in distance_cm_effective_list if distance < whisker_radius])
                false_values = len([distance for distance in distance_cm_effective_list if distance > whisker_radius])
                tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye] = {}
                tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye]['inside (#)'] = true_values
                tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye]['inside (%)'] = str(
                    true_values / (true_values + false_values) * 100) + '%'
                tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye]['outside (#)'] = false_values
                tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye]['outside (%)'] = str(
                    false_values / (true_values + false_values) * 100) + '%'

                mean = np.mean(distance_cm_effective_list)
                std = np.std(distance_cm_effective_list)
                ax[i].hist(distance_cm_effective_list, bins=10, range=(mean - 3 * std, mean + 3 * std),
                           label='{}'.format(len(distance_cm_effective_list)))
                ax[i].set_title(body_part_m2)
                ax[i].legend()
                print(eye, body_part_m2)
                print(tracking_data['body_parts'][body_part_m2]['inside_whisker_range'])
            fig.show()

    return tracking_data,eye_data,mid_estimate_x,mid_estimate_y,r,pix_per_cm,eye_left,eye_right,nose_tip,eye_mid,head_mid

db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M4081']
tracking_data,eye_data,mid_estimate_x,mid_estimate_y,r,pix_per_cm,eye_left,eye_right,nose_tip,eye_mid,head_mid = cli_whisker_analysis(db_path1,mouse1)







def cli_whisker_analysis(db_path,mouse=None,camera_number=6,whisker_radius=3):

    assert camera_number in [5, 6]

    mouse_ids = list(mouse)
    for mouse_id in mouse_ids:

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


        # load the amount of pixels per cm
        pix_per_cm = helpers.get_pixels_per_centimeter(rec_path, video='rpi_camera_' + str(camera_number),marker1='corner_left_left', marker2='corner_lower_right')
        # Then make a lists of left eye, right eye, nose tip, [[x0,y0,likelihood0],[x1,y1,likelihood1],...]list where the likelihood>0.99 for all parts.
        eye_left = np.transpose(np.array([tracking_data['body_parts']['m1_eyecam_left']['position'][:,0]*pix_per_cm,tracking_data['body_parts']['m1_eyecam_left']['position'][:,1]*pix_per_cm,tracking_data['body_parts']['m1_eyecam_left']['likelihood']]))
        eye_right = np.transpose(np.array([tracking_data['body_parts']['m1_eyecam_right']['position'][:, 0]*pix_per_cm,tracking_data['body_parts']['m1_eyecam_right']['position'][:, 1]*pix_per_cm,tracking_data['body_parts']['m1_eyecam_right']['likelihood']]))
        nose_tip = np.transpose(np.array([tracking_data['body_parts']['m1_nose_tip']['position'][:, 0]*pix_per_cm,tracking_data['body_parts']['m1_nose_tip']['position'][:, 1]*pix_per_cm,tracking_data['body_parts']['m1_nose_tip']['likelihood']]))
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

        # look at the cm reference frame and where (x,y) for head center is and what the radius is in pixels
        mid_estimate_x_cm = mid_estimate_x / pix_per_cm
        mid_estimate_y_cm = mid_estimate_y / pix_per_cm
        r = whisker_radius * pix_per_cm

        body_parts_m2 = [body_part for body_part in tracking_data['body_parts'].keys() if 'm2' in body_part.split('_')] #list of m2 body parts
        #fig,ax = plt.subplots(nrows=2,ncols=len(body_parts_m2),figsize=[15,5])

        tracking_data['inside_whisker_range'] = {}
        for j, eye in enumerate(['left', 'right']):
            distance_inside_all_indices = np.array([])
            for i, body_part_m2 in enumerate(body_parts_m2):
                body_part_x_cm = tracking_data['body_parts'][body_part_m2]['position'][:, 0]  # list of x values of bodypart
                body_part_y_cm = tracking_data['body_parts'][body_part_m2]['position'][:, 1]  # list of x values of bodypart
                distance_cm = np.sqrt((body_part_x_cm - mid_estimate_x_cm) ** 2 + (body_part_y_cm - mid_estimate_y_cm) ** 2)  # distance in cm from head center

                distance_cm_closed_eye = np.array([[i,j] for i, j in enumerate(distance_cm) if i in tracking_data['eye_closed_interval'][eye] and np.isnan(j) == False])  # filter out the intervals with closed eyes and the nan values
                distance_inside_per_bodypart = np.array([[i_j[0],i_j[1]] for i_j in distance_cm_closed_eye if i_j[1]<whisker_radius]) #look at the values that are inside the whisker radius
                distance_inside_all_indices = np.append(distance_inside_all_indices,distance_inside_per_bodypart[:,0]) #make a big array where all the indices are in of closed eye intervals
            distance_inside_sorted = np.unique(distance_inside_all_indices) #sort the list and remove double values, so you only have a list with indices at the moments that at least one bodypart is inside the whisker range
            eye_closed_interval_sorted = np.unique(tracking_data['eye_closed_interval'][eye])
            inside = np.array([i for i in tracking_data['eye_closed_interval'][eye] if i in distance_inside_sorted])
            outside = np.array([i for i in tracking_data['eye_closed_interval'][eye] if i not in distance_inside_sorted])
            tracking_data['inside_whisker_range'][eye] = {}
            tracking_data['inside_whisker_range'][eye]['inside / total (#)'] = str(len(inside)) + ' / ' + str(len(inside)+len(outside))
            tracking_data['inside_whisker_range'][eye]['inside (%)'] = str(len(inside)/(len(inside)+len(outside))*100) + '%'
                #Make a plan to get from the list of indices per body part the closest body part each time and then the index when eyes are closed
                #maybe I already wrote this in a function

                #mean = np.mean(distance_cm_effective_list[:, 1])
                #std = np.std(distance_cm_effective_list[:, 1])
                # ax[j][i].hist(distance_cm_effective_list,bins=10,range=(mean-3*std,mean+3*std),label='{}'.format(len(distance_cm_effective_list)))
                # ax[j][i].set_title(body_part_m2)
                # ax[j][i].legend()

                #true_values = len([distance for distance in distance_cm_effective_list if distance < whisker_radius])
                #false_values = len([distance for distance in distance_cm_effective_list if distance > whisker_radius])
                #tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye] = {}
                #tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye]['inside / total (#)'] = str(true_values) + ' / ' + str(true_values+false_values)
                #tracking_data['body_parts'][body_part_m2]['inside_whisker_range'][eye]['inside (%)'] = str(true_values/(true_values+false_values)*100)+'%'

                #print(body_part_m2,(mean-std,mean+std))
                #pprint.pprint(tracking_data['body_parts'][body_part_m2]['inside_whisker_range'])
        #plt.show()
    return tracking_data

db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M3729']
tracking_data = cli_whisker_analysis(db_path1,mouse=mouse1,camera_number=6,whisker_radius=5)
#look at tracking_data['inside_whisker_range'] to see percentage of inside whisker range

