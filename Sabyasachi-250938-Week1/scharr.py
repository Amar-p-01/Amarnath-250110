import cv2
import cv2 as cv
import numpy as np

img = cv.imread("apple.jpg", cv.IMREAD_GRAYSCALE)

scharr = cv.Scharr(img, cv.CV_64F, 1, 0) ** 2 + cv.Scharr(img, cv.CV_64F, 0, 1) ** 2
sobel = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3) **2 + cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3) ** 2

scharr = np.sqrt(scharr)
sobel = np.sqrt(sobel)

scharr /= (np.max(scharr) - np.min(scharr))
sobel /= (np.max(sobel) - np.min(sobel))

cv.imshow("scharr", scharr)
cv.imshow("sobel", sobel)
cv.imshow("diff", np.abs(sobel-scharr))

cv.waitKey(0)