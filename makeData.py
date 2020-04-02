# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:05:43 2017

@author: iwatts
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Load training image and display
img = mpimg.imread('trainingImage.jpg')
plt.imshow(img)

# Convert image from RGB to BGR
cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
# Set number of images to be selected
n = 5

# Loop through and request user input for image selection
print('Select cone images')
for i in range(int(n/2)):
    points = plt.ginput(2)
    cropped = img[int(points[0][1]):int(points[1][1]), int(points[0][0]):int(points[1][0])]
    resized = cv2.resize(cropped, (20, 20)) 
    fileName = 'Cones/cone' + str(i) + '.png'
    cv2.imwrite(fileName,resized)
    i+=1

print('Select not-cone images')
for i in range(n):
    points = plt.ginput(2)
    cropped = img[int(points[0][1]):int(points[1][1]), int(points[0][0]):int(points[1][0])]
    resized = cv2.resize(cropped, (20, 20)) 
    fileName = 'Not-Cones/notCone' + str(i) + '.png'
    cv2.imwrite(fileName,resized)
    i+=1
    