import cv2 as cv 
import numpy as np

img1=cv.imread('/home/keys/Downloads/ed1.jpeg')
img2=cv.imread('/home/keys/Downloads/2.jpeg')

gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

b1=cv.GaussianBlur(gray1,(5,5),0)
b2=cv.GaussianBlur(gray2,(5,5),0)


# using Sobel 

sobelx1=cv.Sobel(gray1,cv.CV_64F,1,0,ksize=5)
sobely1=cv.Sobel(gray1,cv.CV_64F,0,1,ksize=5)



cx1=cv.convertScaleAbs(sobelx1)
cy1=cv.convertScaleAbs(sobely1)


sobel1=cv.addWeighted(cx1,0.5,cy1,0.5,0)

cv.imwrite('sobel1_w.jpg',sobel1)

#using Scharrx

# gx2=cv.Scharr(gray2,cv.CV_64F,1,0)
# gy2=cv.Scharr(gray2,cv.CV_64F,0,1)

# cx2=cv.convertScaleAbs(gx2)
# cy2=cv.convertScaleAbs(gy2)

# scharr2=cv.add(cx2,cy2)

# cv.imwrite('scharr2.jpg',scharr2)

#using Laplacian
# l1=cv.Laplacian(gray1,cv.CV_64F)
# l2=cv.Laplacian(gray2,cv.CV_64F)

# l1=cv.convertScaleAbs(l1)
# l2=cv.convertScaleAbs(l2)

# cv.imwrite('laplacian1.jpg',l1)
# cv.imwrite('laplacian2.jpg',l2)



