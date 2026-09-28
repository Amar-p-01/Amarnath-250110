For the edge detection purpose i used the Sobel, Scharr and Laplacian 

1.for sobel i used the butterfly image . i did the Gausian blur to both of the images for smoothness(Although its optional )
The result initally with cv.add was not that good i added that image with sobel1.jpeg name under the folder week1

For better results i used cv.add Weighted instead of adding the both 100% , which improves the detection much better and boundary are more visible and clear . The result is sobel1_w.jpeg

2.Using the scharr on different image of flower which gives the pretty good results 


3.Then the laplacian on the butterfly image which gives the edges more than the two 