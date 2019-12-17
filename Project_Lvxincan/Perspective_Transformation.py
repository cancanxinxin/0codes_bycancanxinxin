# Perspective Transformation
#coding=utf-8 
'''
to be confirmed to be better
'''

import cv2  
import numpy as np  
from matplotlib import pyplot as plt   

img = cv2.imread('img_wlk.jpg')
rows,cols,ch = img.shape      

pts1 = np.float32([[56,65],[368,52],[28,387],[389,398]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]]) 

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121)
plt.imshow(img)
plt.title('input')
plt.subplot(122)
plt.title('output')
plt.imshow(dst)
plt.show()