# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:06:02 2017

@author: iwatts
"""

import glob
import cv2
import numpy as np
import pickle
from skimage.feature import hog 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import naive_bayes

# Load cone and not-cone image files
filesYes = glob.glob('Cones/*.png')
filesNo = glob.glob('Not-Cones/*.png')
featYes  = np.empty((0,727),dtype='int64')
featNo  = np.empty((0,727),dtype='int64')
windowSize = [20, 20]
# Create ground truth labels
label = np.hstack((np.ones(len(filesYes)),np.zeros(len(filesNo)*3)))

# Loop through cone images and create feature data
for i in filesYes:
    im = cv2.imread(i)
    # Convert image to HSV colorspace
    imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imHSV = cv2.resize(imHSV, tuple(windowSize))
    gray = cv2.resize(gray, tuple(windowSize)) 
    # Compute HOG feature data
    hogFeat = hog(gray, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2))
    # Compute color histogram data
    colorFeatH = cv2.calcHist([imHSV], [0], None, [179], [0,179], False)
    colorFeatS = cv2.calcHist([imHSV], [1], None, [256], [0,256], False)
    colorFeatV = cv2.calcHist([imHSV], [2], None, [256], [0,256], False)
    # Merge feature data
    featInter = np.hstack((hogFeat,colorFeatH.reshape(len(colorFeatH))
                                  ,colorFeatS.reshape(len(colorFeatS))
                                  ,colorFeatV.reshape(len(colorFeatV))))
    # Add feature data to main feature array
    featYes = np.vstack((featYes, featInter))

# Loop through not-cone images and create feature data
for i in filesNo:
    im = cv2.imread(i)
    # Convert image to HSV colorspace
    imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imHSV = cv2.resize(imHSV, tuple(windowSize))
    gray = cv2.resize(gray, tuple(windowSize)) 
    # Compute HOG feature data
    hogFeat = hog(gray, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2))
    # Compute color histogram data
    colorFeatH = cv2.calcHist([imHSV], [0], None, [179], [0,179], False)
    colorFeatS = cv2.calcHist([imHSV], [1], None, [256], [0,256], False)
    colorFeatV = cv2.calcHist([imHSV], [2], None, [256], [0,256], False)
    # Merge feature data
    featInter = np.hstack((hogFeat,colorFeatH.reshape(len(colorFeatH))
                                  ,colorFeatS.reshape(len(colorFeatS))
                                  ,colorFeatV.reshape(len(colorFeatV))))
    # Add feature data to main feature array
    featNo = np.vstack((featNo, featInter))

# Merge cone and not-cone feature data
feat = np.vstack((featYes,featNo,featNo,featNo))
# Normalize feature data
trans = StandardScaler()
featTrans = trans.fit_transform(feat)
# Train classifier
#SVM = naive_bayes.GaussianNB().fit(featTrans,label)
SVM = svm.SVC().fit(featTrans,label)
# Pickle classifier for later use
pickle.dump(SVM,open("classifier.pkl","wb"))
pickle.dump(trans,open("normalizationScalar.pkl","wb"))