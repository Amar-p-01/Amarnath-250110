# Week 1 – Edge Detection

## Approach
Used OpenCV's built-in Sobel and Scharr operators to detect edges by
convolving the grayscale image with derivative kernels.

## Methods compared
- **Sobel X / Y**: detects vertical / horizontal edges respectively.
- **Combined magnitude**:gives overall edge strength.
- **Scharr**: more accurate for small (3x3) kernels than Sobel.

## Observations
- Scharr produced slightly sharper edges than Sobel at ksize=3.
- Grayscale conversion was necessary before applying the kernels.
