# Computer Vision Bootcamp - Week 1 Task 1

I have made a Python script (`edge_detection.py`) performs the following operations:
1. Loads an input image in grayscale.
2. Applies the **Sobel Transform** to compute the edge gradients (using X and Y derivatives).
3. Applies **Canny Edge Detection** after smoothing the image with a Gaussian blur to reduce noise.
4. Generates a side-by-side visual comparison of both edge detection methods using Matplotlib.
5. Automatically creates an `output/` directory and saves the resulting comparison plot as a `.png` file.

## File Structure
```text
Aditi_Kesari_250059_week1/
│
├── task1/
│   └── input_image.jpg         # The source image used for edge detection
│
├── output/
│   └── sample_1_result.png     # The generated output plot (created upon running)
│
├── edge_detection.py           # Main Python script
└── README.md                   # Project documentation
