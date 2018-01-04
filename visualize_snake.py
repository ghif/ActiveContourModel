import image_processor as iproc
import numpy as np

import cv2

# Load snakes history
hist = np.load('results/sn_hist.npy').item()
snake_list = hist['snakes']

# Read input image
Im = cv2.imread('lip.jpg')

for i, S in enumerate(snake_list):
    R = iproc.drawSnakes(Im, S, c=(0, 0, 255))
    R = iproc.normalizeRange(R, minVal=0, maxVal=255)
    impath = "results/snakes.%d.jpg" % i
    cv2.imwrite(impath, R)

