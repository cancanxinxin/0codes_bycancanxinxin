import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('Good Matches&Object detection24.jpg')
cv.circle(img,(1920,480),20,255)

plt.imshow(img)

plt.show()