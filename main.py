import cv2
import numpy as np

img = './writeup/flip/original.jpg'
img = cv2.imread(img)
flip = np.fliplr(img)
cv2.imwrite('./writeup/flip/flipped.jpg', flip)