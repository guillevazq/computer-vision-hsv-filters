import cv2 as cv
import numpy as np
import os
from time import time
from vision import Vision
from hsvfilter import HsvFilter

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the Vision class
vision_limestone = Vision('albion_limestone_processed.jpg')
# initialize the trackbar window
vision_limestone.init_control_gui()

# limestone HSV filter
hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)

image_analyze = cv.imread("images/pic.png")

loop_time = time()
while True:

    # pre-process the image
    processed_image = vision_limestone.apply_hsv_filter(image_analyze)

    # display the processed image
    cv.imshow('Processed', processed_image)

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
