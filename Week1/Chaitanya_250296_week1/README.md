Edge Detection Methods

The given code consists of various methods for edge detection and image processing using Python, OpenCV, and NumPy.

Methods Implemented
1. FILTER2D

Uses the OpenCV filter2D() function to apply custom convolution kernels for image filtering and edge detection.

2. REMOVING NOISE AND THEN DETECTING SCREEN EDGE USING CANNY EDGE DETECTION

Noise is first reduced using image filtering techniques, followed by Canny Edge Detection to identify the edges and boundaries of the screen.

3. SCHARR DERIVATIVES

Uses Scharr operators to calculate horizontal and vertical intensity gradients and detect prominent edges in the image.

4. SOBEL DERIVATIVES

Uses Sobel operators to calculate horizontal and vertical image gradients, which can then be combined to determine the overall edge strength.

5. MANUAL CONVOLUTION

Implements convolution manually using NumPy, demonstrating how image filters and gradient kernels operate at the pixel level without relying entirely on OpenCV's built-in functions.

Technologies Used
Python
OpenCV
NumPy
Google Colab
Objective

The objective of this project is to understand and compare different image filtering, convolution, gradient, and edge detection techniques and observe their performance on real images.