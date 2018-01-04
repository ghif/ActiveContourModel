import image_processor as iproc
import numpy as np
# import active_contour as ac
import snakes 

import skimage.io as  sio
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
from skimage.exposure import equalize_hist

import cv2

# Read image
# Im = cv2.imread('babylip.jpg')
# Im = cv2.imread('koreanlip.jpg')
Im = cv2.imread('lip.jpg')
Img = rgb2gray(Im)
# Img = equalize_hist(Img)

# Initialize contour
## circle
# s = np.linspace(0, 2*np.pi, 400)
# x = 260 + 210 * np.cos(s)
# y = 310 + 210 * np.sin(s)

## ellipse
s = np.linspace(0, 2*np.pi, 400)
x = 210 + 210 * np.cos(s)
y = 90 + 75 * np.sin(s) 

V = np.array([x, y]).T
# V = np.delete(V, -1, axis=0)

R = iproc.drawSnakes(Im, V, c=(255, 0, 0))
# cv2.imshow('footage', R)


# Run Snakes
sn = snakes.Snakes()
S, Edge, Map, Gim = sn.fit_kaas(
        Img,  V, 
        alpha=0.1, beta=3, tau=100,
        w_line=0.0, w_edge=1.0, convergence=0.1
    )

R = iproc.drawSnakes(R, S, c=(0, 0, 255))
# cv2.imshow('footage', R)
# cv2.imshow('Edge', Edge)
# cv2.imshow('Map', Map)

cv2.imwrite('results/snake.jpg', R)
cv2.imwrite('results/map.jpg', iproc.normalizeRange(Map, minVal=0, maxVal=255))
cv2.imwrite('results/gim.jpg', iproc.normalizeRange(Gim, minVal=0, maxVal=255))

# Store snake history
np.save('results/sn_hist.npy', sn.hist)
