# Week 1: Edge Detection Implementation

## Objective
To implement a basic edge detection algorithm using convolutions and kernels.

## Methodology
1. Grayscale Conversion: The image is loaded in grayscale to simplify the gradient calculations (otherwise we'll have to operate on 3D color channels).
2. Kernels: I implemented two kernels:
   - *Sobel Operator*: `3x3` kernel, approximates the derivative of the image while providing some smoothing.
   - *Scharr Operator*: `3x3` kernel, detects fine details and works well in all directions
3. *Convolution*: Applied the kernels using `cv2.filter2D`. `X` kernels detect vertical edges, and `Y` kernels detect horizontal edges.
4. *Magnitude*: Combined the gradients using the Euclidean norm: `sqrt(Gx^2 + Gy^2)`.

## Experimentation & Results
- **Sobel vs. Scharr**: 
  - The Sobel kernel produced clean, well-defined edges and handled noise relatively well due to its center-weighting (`[1, 2, 1]`). 
  - The Scharr kernel (`[3, 10, 3]`) resulted in a significantly higher intensity gradient. It was much more sensitive to subtle texture changes, capturing finer edges but also slightly more noisy compared to Sobel.
- **Conclusion**: Sobel is better for general, distinct object boundaries, while Scharr is preferable when capturing subtle details and textures is required.