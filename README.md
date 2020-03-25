# Template Matching System
In this project I wanted to find a certain images match with my template. In my original templates there is faces in the image so I include face detection in it as well. 

FLANNMatcher.py - 
* takes 2 parameters firs img: image that i want to figure it out whether if fits with my template or not
template: image that defined as template. to understand unknown images are match we use this image as a template image.
* uses FLANN and sift together to find matching template.
* if matching point is more than 100 then it is a match, otherwise it is not a match.

edgeDetection.py - 
* uses canny to finds edge of the image 
* takes only 1 parameter which is an image

faceDetection.py -
* find face on an image
* takes only 1 parameter which is an image

 
