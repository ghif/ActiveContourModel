import numpy as np
import cv2

def normalizeRange(Img, minVal=0, maxVal=255):
    srcMinVal = np.min(Img)
    srcMaxVal = np.max(Img)

    # rescale
    ds = srcMaxVal -srcMinVal
    dt = maxVal - minVal

    # Normalize
    Out = (Img - srcMinVal) * 1.0
    Out = dt * Out / ds

    return Out

def drawSnakes(Img, S, c=(255, 0, 0)):
    R = Img.copy()
    
    for s in S:
        [v1, v2] = s
        v1 = int(v1)
        v2 = int(v2)

        cv2.circle(R, (v1, v2), 3, c)

    return R
