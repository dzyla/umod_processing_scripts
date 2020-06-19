'''

Script used for joining the PyMOL png frames into a movie

'''

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import glob

movie_frames = glob.glob(r'path/to/images/*.*')

movie_side = []
movie_top = []
for frame in movie_frames:

    if 'side' in frame:

        frame_data = plt.imread(frame)
        movie_side.append(frame_data)

    elif 'top' in frame:

        frame_data = plt.imread(frame)
        movie_top.append(frame_data)

movie_top = np.stack(movie_top)
movie_side = np.stack(movie_side)

movie = np.concatenate((movie_side, movie_top), axis=1)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (500,100)
fontScale              = 2
fontColor              = (0,0,0)
lineType               = 0


position_fig = (280,100)
position_label1 = (560,620)
position_label2 = (560,1250)

movie_rgb = []


for n, frame in enumerate(movie):


    frame = cv2.putText(
     frame, #numpy array on which text is written
     "Flexibility of UMOD filament", #text
     position_fig, #position at which writing has to start
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     1.5, #font size
     (0, 0, 0, 255), #font color
     3) #font stroke

    frame = cv2.putText(
     frame, #numpy array on which text is written
     "Side view", #text
     position_label1, #position at which writing has to start
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     1, #font size
     (0, 0, 0, 255), #font color
     2) #font stroke

    frame = cv2.putText(
     frame, #numpy array on which text is written
     "Top view", #text
     position_label2, #position at which writing has to start
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     1, #font size
     (0, 0, 0, 255), #font color
     2) #font stroke

    frame  = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    movie_rgb.append(frame)

plt.imshow(movie_rgb[1])
plt.show()

movie = movie_rgb

movie_name = 'movie_ZPC.mp4'

imageio.mimwrite(movie_name, movie , fps = 5)