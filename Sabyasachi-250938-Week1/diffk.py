import cv2
import cv2 as cv
import numpy as np

img = cv.imread("tyrion.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow("img", img)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=17)

sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=17)


magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

magnitude /= np.max(magnitude)
cv.imshow("raw_mag k1", magnitude)
edges = (magnitude > 0.2).astype(float)
cv.imshow("edges k1", edges)


sobel_x = cv2.Sobel(img, cv2.CV_64F, 3, 0, ksize=17)

sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 3, ksize=17)


magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

magnitude /= np.max(magnitude)
cv.imshow("raw_mag k3", magnitude)
edges = (magnitude > 0.2).astype(float)
cv.imshow("edges k3", edges)

cv.waitKey(0)