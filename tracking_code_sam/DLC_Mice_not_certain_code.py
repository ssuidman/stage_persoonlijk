M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_0 = []
for i in M3728_alone_cam5_mice_low_likelihood_reduced_frames_index[0]:
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_0.append(M3728_alone_cam5_mice_likelihood[M3728_alone_cam5_mice_likelihood_columns[0]][i])



######
M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood = [] #list where all reduced likelihoods for all bodyparts come in
for j in range(len(M3728_alone_cam5_mice_low_likelihood_reduced_frames_index)): #getting across all bodypart-indices (0-14)
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_column = [] #likelihood list per bodypart
    for i in M3728_alone_cam5_mice_low_likelihood_reduced_frames_index[j]: #look at a specific list with subsequent indices in a bodypart list
        subsequent_likelihood = M3728_alone_cam5_mice_likelihood[M3728_alone_cam5_mice_likelihood_columns[j]][i]
        M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_column.append()
    M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood.append(M3728_alone_cam5_mice_low_likelihood_reduced_frames_likelihood_column)
