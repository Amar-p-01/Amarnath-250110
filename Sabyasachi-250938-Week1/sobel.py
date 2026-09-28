import cv2
import cv2 as cv
import numpy as np

img = cv.imread("tyrion.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow("img", img)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=17)
cv.imshow("sobel_x (scaled)", sobel_x/(np.max(sobel_x) - np.min(sobel_x)))

sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=17)
cv.imshow("sobel_y (scaled)", sobel_y/(np.max(sobel_y) - np.min(sobel_y)))


magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

magnitude /= np.max(magnitude)
cv.imshow("raw_mag", magnitude)
edges = (magnitude > 0.2).astype(float)
cv.imshow("edges", edges)
cv.waitKey(0)