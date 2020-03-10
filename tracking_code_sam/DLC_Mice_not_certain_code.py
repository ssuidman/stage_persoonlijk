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



####
M3728_alone_cam5_mice_low_likelihood_all_frames_try = [] #second way to get all low likelihood frames, but with the names with them
for i in M3728_alone_cam5_mice_likelihood_columns:
    M3728_alone_cam5_mice_low_likelihood_all_frames_try.append(M3728_alone_cam5_mice_likelihood[i][M3728_alone_cam5_mice_likelihood[i]<0.99])
#and to get the index:
M3728_alone_cam5_mice_low_likelihood_all_frames_try[0].index #index of the low likelihoods
M3728_alone_cam5_mice_low_likelihood_all_frames_try[0].array #array of pandas array
M3728_alone_cam5_mice_low_likelihood_all_frames_try[0].name #name of the bodypart






######
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
    plt.legend()
plt.savefig('/Users/samsuidman/Desktop/plaatje.png',dpi=1200) #saving the picture at high quality


