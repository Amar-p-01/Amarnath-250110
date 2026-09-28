import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

# Loading image
img = cv.imread('/content/20260821_162727.jpg')
cv2_imshow(img)

#convert into grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Sobel kernels
# vertical edges
sobel_x = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0, ksize=3)
# horizontal edges
sobel_y = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=3) 

# Convert back 
abs_sobel_x = cv.convertScaleAbs(sobel_x)
abs_sobel_y = cv.convertScaleAbs(sobel_y)

# Combine x and y gradients into one edge map
combined = cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


scharr_x = cv.Scharr(gray, cv.CV_64F, dx=1, dy=0)
scharr_y = cv.Scharr(gray, cv.CV_64F, dx=0, dy=1)
scharr_x_display = cv.convertScaleAbs(scharr_x)
scharr_y_display = cv.convertScaleAbs(scharr_y)
scharr_combined = cv.addWeighted(scharr_x_display, 0.5, scharr_y_display, 0.5, 0)

cv2_imshow(scharr_x_display)
cv2_imshow(scharr_y_display)
cv2_imshow(scharr_combined)
# Display all results
cv2_imshow(gray)
cv2_imshow(abs_sobel_x)   
cv2_imshow(abs_sobel_y)   
cv2_imshow(combined)     
