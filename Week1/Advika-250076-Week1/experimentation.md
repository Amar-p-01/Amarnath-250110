# Edge Detection Experimentation

# What this code does
For this task, I wrote a Python script to detect edges in an image using OpenCV's built-in tools to apply the Sobel & Scharr operator in the image.

# Methods Tested
1. Loaded the test image in black-and-white.
3. Used `cv2.Sobel()` to scan the image for edges in two directions: up-and-down (vertical) and left-to-right (horizontal)
4. Merged the horizontal and vertical edges using `cv2.magnitude()`. This calculates the total edge strength using the formula: $G = \sqrt{G_x^2 + G_y^2}$.
5. Finally, I used `cv2.normalize()` to scale the math results back into standard image pixels (0 to 255) so the final edge map could be displayed clearly on screen.

# Conclusion
The Sobel operator is the most practical choice for basic edge detection on standard images due to its better noise-smoothing characteristics . 
