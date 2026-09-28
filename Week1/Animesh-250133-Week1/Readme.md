Image preprocessing: image loading, resizing, BGR → grayscale conversion, pixel-level operations.

Convolution from scratch: implemented sliding-window convolution with custom kernels to understand filtering at the pixel level.

Custom kernels: experimented with smoothing/blur, vertical-edge, and horizontal-edge filters.

OpenCV filtering: used cv2.filter2D() and compared built-in filtering with the manual implementation.

Edge detection: implemented and compared Sobel, Scharr, and Canny, including gradient computation and edge magnitude.

Output processing: handled signed gradient values using cv2.convertScaleAbs() and normalization for visualization.

Comparison & experimentation: observed how kernel design, operator choice, and Canny threshold values affect the resulting edge maps.