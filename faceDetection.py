import cv2
import sys

cascPath = sys.argv[1]
face_cascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread('cropedOne.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image, 1.3, 5)
for (x,y,w,h) in faces:
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()