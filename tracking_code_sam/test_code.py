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