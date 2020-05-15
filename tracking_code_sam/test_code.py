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
                var_high_sequences_compressed_per_bodypart_temp.extend(var_high_sequences_per_bodypart[j].index) #you merge the sequences that should be together
            var_high_sequences_compressed_per_bodypart.append(var_high_sequences_compressed_per_bodypart_temp) #you add this list of compressed sequences to the big list
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







def func_speed(tracking_data):
    body_parts = tracking_data['body_parts'].keys() #make a list for the body parts
    for body_part in body_parts: #look at one bodypart
        position = tracking_data['body_parts'][body_part]['position'] #get the position data for this body_part
        position_difference = np.transpose(np.diff(np.transpose(position))) #calculate the difference vector of position. This means that this array has one less element.
        velocity_not_same_size = np.transpose([np.transpose(position_difference)[c] / np.diff(tracking_data['timestamps']) for c in range(len(np.transpose(position_difference)))]) #calculate the velocity vector by dividing through the time between two positions
        velocity = np.transpose([np.insert(np.transpose(velocity_not_same_size)[c],0,np.nan) for c in range(len(np.transpose(velocity_not_same_size)))]) #add np.nan in front, so that the first element of the velocity vector is empty. And the second element is the velocity relative to the previous position
        speed = np.asarray([np.sqrt(c[0]**2 +c[1]**2) for c in velocity]) #calculate the speed using the velocity
        speed_averaged = [np.sum(speed[c-5:c+5])/(5*2+1) for c in range(len(speed))] #averaged over 5 frames before and after
        tracking_data['body_parts'][body_part]['speed'] = speed #add the speed to the dictionary of tracking data
        tracking_data['body_parts'][body_part]['speed_averaged'] = speed_averaged
    return tracking_data





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

        tracking_data = func_speed(tracking_data)

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
    return tracking_data, eye_data




db_path1 = "/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database"
mouse1 = ['M4081']
tracking_data1,eye_data1 = cli_align_egocentric(db_path1,mouse1)

#This is a list of lists (can be made into an array if needed), where the intervals of closed (left) eye are in.
eye_closed_interval_left_full = [list(range(eye_data1['left']['eye_closed_interval'][c][0],eye_data1['left']['eye_closed_interval'][c][1]+1)) for c in range(len(eye_data1['left']['eye_closed_interval']))]


fig,ax = plt.subplots(ncols=len(body_parts),figsize=[32.4,4.8])
for i in range(len(tracking_data1['body_parts'].keys())):
    ax[i].plot(tracking_data1['body_parts'][list(tracking_data1['body_parts'].keys())[i]]['speed_averaged'])
    ax[i].set_title(list(tracking_data1['body_parts'].keys())[i])
fig.tight_layout()
fig.show()

#There is averaged over 11 (=5*2+1) frames (func_speed) and this is plotted then.

# cam3, cam4 --> fps=60
# cam5, cam6 --> fps=30