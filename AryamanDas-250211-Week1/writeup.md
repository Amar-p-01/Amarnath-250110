## Edge detection implementation

- Tried Sobel and Scharr kernels for edge detection. Scharr has heavier weights and is more sensitive to finer details. However the difference isn't really visible so I separately showed the diff.
- Also applied Gaussian blur before Sobel/Scharr to reduce noise.