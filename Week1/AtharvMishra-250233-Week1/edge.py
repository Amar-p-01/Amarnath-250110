import cv2
import numpy as np
import matplotlib.pyplot as plt

img  = cv2.imread('data/paisededopilijjjj.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

horizontal = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
vertical   = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobbelled  = np.sqrt(horizontal**2 + vertical**2)
horizontal = np.uint8(255 * np.abs(horizontal) / np.max(horizontal))
vertical   = np.uint8(255 * np.abs(vertical) / np.max(vertical))
sobbelled  = np.uint8(255 * sobbelled / np.max(sobbelled))

cv2.imshow('Original', img)
cv2.imshow('horizontal', horizontal)
cv2.imshow('vertical', vertical)
cv2.imshow('sobbelled', sobbelled)
cv2.waitKey(0)
cv2.destroyAllWindows()