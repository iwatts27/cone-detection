# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:40:34 2017

@author: iwatts
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from skimage.feature import hog 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from scipy.ndimage.measurements import label

# Load trained classifier
classifier = pickle.load(open("classifier.pkl", 'rb'))
normScalar = pickle.load(open("normalizationScalar.pkl", 'rb'))
# Load image
img = mpimg.imread('testImage.jpg')
# Convert image to HSV colorspace
imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

scrollImg = []
posWindows = []
windowList = []
featTest = []
trans = StandardScaler()
xStartStop = [0,img.shape[1]]
yStartStop = [0,img.shape[0]]
# Set window size for sliding search
windowSize = [20, 20]
# Set window overlap percentage
overlap = [0.75, 0.75]

plt.figure(2)
plt.clf()
plt.imshow(img)
# Request mouse input for four points to define boundaries of the arena
# Start from desired origin and move clockwise around the arena
source = np.asarray(plt.ginput(4),np.float32)
t = time.time()
dest = np.array([[0,400],[0,0],[400,0],[400,400]],np.float32)

# Compute the perspective transform
transform = cv2.getPerspectiveTransform(source, dest)
warped = cv2.warpPerspective(img,transform,(400,400))

# Display warped persepective image of arena
plt.figure(3)
plt.imshow(warped)

# Compute the span of the region to be searched
xspan = xStartStop[1] - xStartStop[0]
yspan = yStartStop[1] - yStartStop[0]

# Compute the number of pixels per step in x/y
nxPixPerStep = np.int(windowSize[0]*(1 - overlap[0]))
nyPixPerStep = np.int(windowSize[1]*(1 - overlap[1]))

# Compute the number of windows in x/y
nxWindows = np.int(xspan/nxPixPerStep) - 2
nyWindows = np.int(yspan/nyPixPerStep) - 2

# Loop through finding x and y window positions
for ys in range(nyWindows):
	for xs in range(nxWindows):
		# Calculate window position
		startx = xs*nxPixPerStep + xStartStop[0]
		endx = startx + windowSize[0]
		starty = ys*nyPixPerStep + yStartStop[0]
		endy = starty + windowSize[1]
		# Append window position to list
		windowList.append(((startx, starty), (endx, endy)))
# Loop through windows and generate features
imgFeat = np.empty((len(windowList),727),dtype='int64')
i = 0
for window in windowList:
    # Crop window from full images
    scrollImg = imgHSV[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    scrollImgGray = gray[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    scrollImg = cv2.resize(scrollImg, tuple(windowSize))
    scrollImgGray = cv2.resize(scrollImgGray, tuple(windowSize)) 
    # Compute HOG and color features
    hogFeat = hog(scrollImgGray, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2))
    colorFeatH = cv2.calcHist([scrollImg], [0], None, [179], [0,179], False)
    colorFeatS = cv2.calcHist([scrollImg], [1], None, [256], [0,256], False)
    colorFeatV = cv2.calcHist([scrollImg], [2], None, [256], [0,256], False)
    # Merge features
    featInter = np.hstack((hogFeat, colorFeatH.reshape(len(colorFeatH))
                                  , colorFeatS.reshape(len(colorFeatS))
                                  , colorFeatV.reshape(len(colorFeatV))))
    # Add to array of sample features
    # Was using vstack to do this but doing it this way sped the loop up over 2000%
    imgFeat[i][:] = featInter
    i += 1

# Normalize feature data
imgFeatTrans = normScalar.transform(imgFeat)
# Create predictions from feature data
prediction = classifier.predict(imgFeatTrans)

# Create heat map
heatMap = np.zeros_like(gray)
# Loop through windows and determine if predicted a cone
for i in range(len(windowList)):
    if prediction[i] == 1:
        heatMap[windowList[i][0][1]:windowList[i][1][1], 
                windowList[i][0][0]:windowList[i][1][0]] += 1
# Set lower limit on heat map
heatMap[heatMap < 2] = 0
labels = label(heatMap)
# Plot heat map
plt.figure(4)
plt.clf()
plt.imshow(labels[0])

imgMod = img[:]
# Loop through creating rectangles and positional text for cones
for i in range(1,int(labels[1]+1)):
    points = np.argwhere(labels[0]==i)
    x = []
    y = []
    for point in points:
        x.append(point[1])
        y.append(point[0])
    # Define vertices of the rectangle
    UL = [min(x), min(y)]
    LR = [max(x), max(y)]
    # Compute width and height of the rectangle
    w = LR[0] - UL[0]
    h = LR[1] - UL[1]
    # Compute lower center point of the rectangle
    LC = np.array([int((min(x)+max(x))/2), max(y)],np.float32)
    LCi = np.array([np.array([LC])])
    LCTrans = cv2.perspectiveTransform(LCi,transform)
    # Throw out rectangles that are too small, too large, or not in the arena
    if LCTrans[0][0][0] < 400 and LCTrans[0][0][0] > 0 and LCTrans[0][0][1] < 400 and LCTrans[0][0][1] > 0 and w < 100 and w > 25 and h < 100 and h > 25:
        # Draw rectangle
        cv2.rectangle(imgMod, (UL[0],UL[1]), (LR[0],LR[1]), (0,0,0), 2)
        # Compute distance from arena origin
        xFt = round(LCTrans[0][0][0] * .03, 1)
        yFt = round((400 -LCTrans[0][0][1]) * .03, 1)
        # Draw text
        text = '(' + str(xFt) + ',' + str(yFt) + ')'
        cv2.putText(imgMod,text, (LC[0],LC[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
        
# Plot modified images with text and rectangles
plt.figure(5)
plt.clf()
plt.imshow(imgMod)

t2 = time.time()
print('Run time (s):', round(t2-t, 2))