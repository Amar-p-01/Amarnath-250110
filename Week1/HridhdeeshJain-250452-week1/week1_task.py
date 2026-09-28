import cv2
import numpy as np

image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray,cv2.CV_64F, 1,0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F,0, 1,ksize=3)
abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)
sobel = cv2.add(abs_sobelx, abs_sobely)
cv2.imwrite('sobelig.jpg',sobel)

scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
abs_scharrx = cv2.convertScaleAbs(scharrx)
abs_scharry = cv2.convertScaleAbs(scharry)
scharr = cv2.add(abs_scharrx, abs_scharry)
cv2.imwrite('scharrig.jpg',scharr)

laplace = cv2.Laplacian(gray, cv2.CV_64F)
laplace = cv2.convertScaleAbs(laplace)
cv2.imwrite('laplaceig.jpg', laplace)


