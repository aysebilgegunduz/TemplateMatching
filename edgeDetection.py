import cv2
import numpy as np
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# construct the argument parse and parse the arguments
# loop over the images
#for imagePath in glob.glob(args["images"] + "/*.jpg"):
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread('images/original.jpg')
#image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

auto = cv2.resize(auto, (960, 540))
tight = cv2.resize(tight, (960, 540))
wide = cv2.resize(wide, (960, 540))

# show the images
#cv2.imshow("Original", image)
#cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.imshow("Edges", wide)
cv2.waitKey(0)

