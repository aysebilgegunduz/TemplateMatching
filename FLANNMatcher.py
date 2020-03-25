import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
imgPath = ""
templatePath=""
for i in glob.glob(imgPath+"*.png"):
    img = cv2.imread(i,0)
    template = cv2.imread(templatePath,0) #queryImage

    sift = cv2.xfeatures2d.SIFT_create()

    #Find Keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img, None)

    #FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(template,kp1,img,kp2,matches,None,**draw_params)
    counter = 0
    for i in range(len(matchesMask)):
        if matchesMask[i] != [0,0]:
            counter += 1
            
    print(counter)
    if counter > 100:
        print("this photo fits with my template")
        plt.imshow(img3, ), plt.show()
    else:
        print("This photo didn't fit with my template")
        os.remove(i)
