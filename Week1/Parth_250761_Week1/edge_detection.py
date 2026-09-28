# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

image= cv2.imread("C:/Users/paart/Downloads/devta.png")
gray2= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray= cv2.equalizeHist(gray2)
blurred_image = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
blurred_image2 = cv2.GaussianBlur(gray2, (5, 5), sigmaX=0)

plt.imshow(blurred_image, cmap= "gray")
plt.axis("off")
plt.show()

# %%
sobel_x= np.array([
[-1,0, 1],
[-2,0,2],
[-1,0,1]
])
sobel_y= np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])

laplacian= np.array([
    [0,2,0],
    [2,-8,2],
    [0,2,0]
])

def convolution(img, kernel, stride):
    h,w= img.shape
    kh,kw= kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.pad(
            img,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0
        )
    output_height = ((h + 2 * pad_h - kh) // stride) + 1
    output_width = ((w + 2 * pad_w - kw) // stride) + 1

    output= np.zeros((output_height,output_width),dtype=float)

    for i in range(output_height):
     for j in range(output_width):
        row= (i*stride) 
        col=(j*stride)
        region= padded[
           row:row+kh,
           col:col+kw
        ]
        output[i,j]= np.sum(region*kernel)


    return output

       
laplace= convolution(blurred_image2,laplacian,1)
plt.imshow(np.abs(laplace),cmap="gray")
plt.axis("off")
plt.show()

edge_y= convolution(blurred_image, sobel_y,1)
plt.imshow(np.abs(edge_y),cmap="grey")
plt.axis("off")


edge_x= convolution(blurred_image,sobel_x,1)
plt.imshow(np.abs(edge_x),cmap="grey")
plt.axis("off")


sobel_final= np.sqrt(edge_x**2 + edge_y**2)
sobel_final= cv2.normalize(sobel_final,None,0,255,cv2.NORM_MINMAX )
plt.imshow(np.abs(sobel_final),cmap="grey") 
plt.axis("off")
plt.show()
