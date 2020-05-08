import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import time

t1 = time.time()



#reading the tracking files (h5), these variables now contain a matrix, 3 elements per
#bodypart (x,y,likelihood). There are 10 bodyparts, so in total there are 30 elements.
#Each element got almost 20000 values, which are coordinate-values (for x and y) or
#likelihoodvalues
M3728_alone_cam5_box = pd.read_hdf('/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_5DLC_resnet50_M3728_boxFeb14shuffle1_500000.h5')
M3728_alone_cam5_mice = pd.read_hdf('/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5')
M3728_alone_cam6_box = pd.read_hdf('/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_6DLC_resnet50_M3728_boxFeb14shuffle1_500000.h5')
M3728_alone_cam6_mice = pd.read_hdf('/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_6DLC_resnet50_M3728_miceFeb14shuffle1_1030000.h5')



#make a list of likelihood column names (this is a vector)
M3728_alone_cam5_mice_likelihood_columns = []
for i in M3728_alone_cam5_mice.columns:
    if i[2] == 'likelihood':
        M3728_alone_cam5_mice_likelihood_columns.append(i)
# make a list of likelihoodvalues per tracking object (so this is a matrix)
M3728_alone_cam5_mice_likelihood = M3728_alone_cam5_mice[M3728_alone_cam5_mice_likelihood_columns]




#This saves figures of the likelihoodp plot (xlabel=which frame , ylabel=likelihoodvalue (0-1)
#for i in M3728_alone_cam5_mice_likelihood_columns:
#    plt.figure()
#    plt.plot(M3728_alone_cam5_mice_likelihood[i])
#    title = "M3728_alone_cam5_mice_{}_{}".format(i[2],i[1])
#    plt.title(title)
#    plt.savefig("/Users/samsuidman/Desktop/likelihood_figures/{}".format(title))





#with ... as ...: opens and closes a file, this is nice because otherwise the files stays opened.
with imageio.get_reader("/Users/samsuidman/Desktop/files_from_computer_arne/shared_data/social_interaction_eyetracking/database/M3728/2019_07_09/social_interaction/2019-07-09_14-55-57_Silence_box_no_enclosure/rpi_camera_5DLC_resnet50_M3728_miceFeb14shuffle1_1030000_labeled.mp4") as M3728_alone_cam5_mice_video:
    M3728_alone_cam5_mice_video_meta_data = M3728_alone_cam5_mice_video.get_meta_data() #this contains the metadata, such as  fps (frames per second), duration, etc.
    M3728_alone_cam5_mice_video_frames = M3728_alone_cam5_mice_video.count_frames() #counting the amount of frames (=19498)
#    M3728_alone_cam5_mice_video_dataframe_2364 = M3728_alone_cam5_mice_video.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 19497





M3728_alone_cam5_mice_low_likelihood_all_frames = [] #making a list where all low likelihoods indices (which frame is it. Example: frame 11111) of all bodyparts come
likelihood_threshold = 0.99 #setting the likelihood threshold (starting with a conservative 0.99)
for j in M3728_alone_cam5_mice_likelihood_columns: #get the name of a likelihood column
    M3728_alone_cam5_mice_low_likelihood_all_frames_per_column = [] #making a list where the indices of low likelihood frames of a specific body part are in
    k = 0 #start counting the indices at 0
    for i in M3728_alone_cam5_mice_likelihood[j]: #go through all the low likelihood values of a column
        if i < likelihood_threshold: #see if the value if lower than the likelihood threshold (starting with 0.99)
            M3728_alone_cam5_mice_low_likelihood_all_frames_per_column.append(k) #append the index if i<0.99
        k+=1
    M3728_alone_cam5_mice_low_likelihood_all_frames.append(M3728_alone_cam5_mice_low_likelihood_all_frames_per_column) #append the list with low likelihoods of a certain column to the list with low likelihoods for all columns



#To plot bodypart i (0-14) you can do:
#plt.plot(M3728_alone_cam5_mice_low_likelihood_all_frames[i])
#title = "{}".format(M3728_alone_cam5_mice_likelihood_columns[i][1])
#plt.title(title)
#plt.show()





M3728_alone_cam5_mice_low_likelihood_reduced_frames_index = [] #making a list for all body parts, where the indices of frames with low likelihood are in, but reduced so that subsequent frames are not in it
for i in range(len(M3728_alone_cam5_mice_likelihood_columns)): #go through all body parts
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_column = [] #making a list for one body part. Several lists of indices of subsequent low likelihoods are stored in this bigger list
    for first_low_likelihood_frame in M3728_alone_cam5_mice_low_likelihood_all_frames[i]: #go through the list of all low likelihood frames
        if first_low_likelihood_frame-1 not in M3728_alone_cam5_mice_low_likelihood_all_frames[i]: #this is so that no lists are made that contain the same elements as another list
            second_variable = first_low_likelihood_frame #set a second variable to avoid conflict with the first one
            subsequent_list = [] #making a list where subsequent low likelihood indices can be stored in
            subsequent_list.append(second_variable) #append the first low likelihood value to this list
            while second_variable+1 in M3728_alone_cam5_mice_low_likelihood_all_frames[i]: #if the subsequent index of the low likelihood list is also in the low likelihood list, then look at this next index
                second_variable+=1 #look at the next index
                subsequent_list.append(second_variable) #add the next index also to the list
            M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_column.append(subsequent_list) #add the list with subsequent indices to the list for one bodypart.
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_index.append(M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_column) #add the list for one bodypart to the list for all bodyparts
#Now there is a list "M3728_alone_cam5_mice_low_likelihood_reduced_frames_index". For example "M3728_alone_cam5_mice_low_likelihood_reduced_frames_index[3][5:8]" gives now for the third bodypart: the fourth till the seventh subsequent low likelihood frames in lists .





M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood = [] #list where all reduced likelihoods for all bodyparts come in
for j in range(len(M3728_alone_cam5_mice_low_likelihood_reduced_frames_index)): #getting across all bodypart-indices (0-14)
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_column = [] #likelihood list per bodypart
    for i in M3728_alone_cam5_mice_low_likelihood_reduced_frames_index[j]: #look at a specific list with subsequent indices in a bodypart list
        subsequent_likelihood = M3728_alone_cam5_mice_likelihood[M3728_alone_cam5_mice_likelihood_columns[j]][i] #make a small list of likelihoods that match the small list of indices
        M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_column.append(subsequent_likelihood) #add this small likelihood list to the list of one body part
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood.append(M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_column) #add the one body part list to the list of all bodyparts
#Now there is a list that matches the list "M3728_alone_cam5_mice_low_likelihood_reduced_frames_index", but then with all the likelihood values.
# To look at a value with the index "M3728_alone_cam5_mice_low_likelihood_reduced_frames_index[3][5][0]", you have to type "M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood[3][5].array[0]",
# or without the [0] after array, it just gives an array of the likelihood. "M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood[3][5].index[0]" has the same value as the indexlist that was created
#Conclusion:
#           M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood[body_part_index(0-14)][number_of_sequence(~39)].array
#           M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood[body_part_index(0-14)][number_of_sequence(~39)].index
#are the outputs that you can use




M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_single_frame = [] #Making a list (within it a list) from the indices where a sequence has a minimal likelihood
M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_single_frame = [] #Making the list with corresponding likelihoods
for j in M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood: #look at the 15 bodyparts
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_single_frame_column = [] #look at index per bodypart
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_single_frame_column = [] #look at the likelihood per bodypart
    for i in j: #look at the list-sequences per bodypart
        index_min = i.idxmin() #look at the index corresponding to the minimal likelihood in a sequence
        likelihood_min = min(i.array) #looking at the minimal likelihood of a sequence
        M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_single_frame_column.append(index_min) #add the index to the index_list per bodypart
        M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_single_frame_column.append(likelihood_min) #add the likelihood to the likelihood_list per bodypart
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_single_frame.append(
        M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_single_frame_column) #add each index bodypart list to the big index list
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_single_frame.append(
        M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_single_frame_column) #add each likelihood bodypart list ot the big likelihood bodypart list
#Conclusion:
#       There are 2 lists "M3728_alone_cam5_mice_low_likelihood_reduced_frames_index_single_frame" and
#       "M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_single_frame". These lists
#       contain 15 elements each (for each body part one list). The lists per bodypart contain the lowest values
#       of likelihood per sequence (low likelihood sequence with following frames that have p<0.99).




M3728_alone_cam5_mice_likelihood_binary = [] #Making a list for all bodyparts (len(list)=15) where if p<0.99 the value of list[i] becomes NaN and otherwise becomes 1+i (so that the lines are above eachother)
k=0 #place for the first line
for i in M3728_alone_cam5_mice_likelihood_columns: #going through each bodypart
    M3728_alone_cam5_mice_likelihood_binary_column = [] #making a list per bodypart where [1,NaN,NaN,1,1,...]
    for j in M3728_alone_cam5_mice_likelihood[i]: #cloning the likelihood list of a bodypart [0.3338,0.9783,...]
        M3728_alone_cam5_mice_likelihood_binary_column.append(j) # " " "
    M3728_alone_cam5_mice_likelihood_binary_column = np.asarray(M3728_alone_cam5_mice_likelihood_binary_column) #making type=array of it
    M3728_alone_cam5_mice_likelihood_binary_column[M3728_alone_cam5_mice_likelihood_binary_column<likelihood_threshold] = np.NaN #setting p<0.99 to not a number
    M3728_alone_cam5_mice_likelihood_binary_column[M3728_alone_cam5_mice_likelihood_binary_column>likelihood_threshold] = len(M3728_alone_cam5_mice_likelihood_columns)-k #setting p>0.99 to 15 (or 14,13,...,2,1) so that all lines can be plotted above eachother)
    M3728_alone_cam5_mice_likelihood_binary.append(M3728_alone_cam5_mice_likelihood_binary_column) #adding the list of one bodypart to the bigger list
    k+=1 #setting k+1 so that the next bodypart comes at the line above/below the other

for j in range(len(M3728_alone_cam5_mice_likelihood_columns)): #going through each body part
    plt.plot(M3728_alone_cam5_mice_likelihood_binary[j], label="{}".format(M3728_alone_cam5_mice_likelihood_columns[j][1])) #plot the line of the binary likelihood
    plt.legend() #making the legend
plt.savefig('/Users/samsuidman/Desktop/likelihood_figures/plaatje.png',dpi=1200) #saving the picture at high quality







t2 = time.time()

print('The time it took for running is {}'.format(t2-t1))


#the script works if I enter everything above here at once (and takes about 63 seconds to run)

