Tried out sobel and scharr kernels on some images

### method: 
find sobel x and y, scale and display individually. 
Then find all edges with root(sobelx^2 + sobely^2), scale adn display

### experimentation:

Tried out different kernel sizes and both sobel and scharr. 

Larger kernels resulted in less edges being detected (only stronger edges survived)

Sobel Scharr mei no diff almost somehow

Also played with dx and dy, interesting effects - apparently dx controls which order derivative is used
