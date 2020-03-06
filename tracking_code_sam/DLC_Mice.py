import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio

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
    M3728_alone_cam5_mice_video_meta_data = M3728_alone_cam5_mice_video.get_meta_data()
    M3728_alone_cam5_mice_video_frames = M3728_alone_cam5_mice_video.count_frames()
#    M3728_alone_cam5_mice_video_dataframe_2364 = M3728_alone_cam5_mice_video.get_data(2364) #this contains the data from the 2364's frame. The max number between brackets is in this case 19497


likelihood_threshold = 0.99 #setting the likelihood threshold (starting with a conservative 0.99)
M3728_alone_cam5_mice_low_likelihood_all_frames = [] #making a list where all low likelihoods indices (which frame is it. Example: frame 11111) of all bodyparts come
for j in M3728_alone_cam5_mice_likelihood_columns: #get the name of a likelihood column
    M3728_alone_cam5_mice_low_likelihood_all_frames_per_column = [] #making a list where the indices of low likelihood frames of a specific body part are in
    k = 0 #start counting the indices at 0
    for i in M3728_alone_cam5_mice_likelihood[j]: #go through all the low likelihood values of a column
        if i < likelihood_threshold: #see if the value if lower than the likelihood threshold (starting with 0.99)
            M3728_alone_cam5_mice_low_likelihood_all_frames_per_column.append(k) #append the index if i<0.99
        k+=1
    M3728_alone_cam5_mice_low_likelihood_all_frames.append(M3728_alone_cam5_mice_low_likelihood_all_frames_per_column) #append the list with low likelihoods of a certain column to the list with low likelihoods for all columns
