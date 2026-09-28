import os
import cv2

img_path = os.path.join('.', 'bnd.jpg')
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) 
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 

sobel_combined = cv2.magnitude(sobel_x, sobel_y)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

cv2.imshow('Original', img)
cv2.imshow('Grayscale', gray)
cv2.imshow('Sobel X (vertical edges)', sobel_x)
cv2.imshow('Sobel Y (horizontal edges)', sobel_y)
cv2.imshow('Sobel Combined', sobel_combined)

scharr_x = cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_64F, 1, 0))
scharr_y = cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_64F, 0, 1))

cv2.imshow('Scharr X', scharr_x)
cv2.imshow('Scharr Y', scharr_y)


cv2.waitKey(0)
cv2.destroyAllWindows()
