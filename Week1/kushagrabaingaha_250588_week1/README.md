# Edge Detection Using OpenCV

## Objective

To detect horizontal and vertical edges in the provided cat image using convolution with predefined Sobel kernels.

## Method

The image was loaded with OpenCV and converted from colour to grayscale. Two 3×3 Sobel kernels were then applied using `cv2.filter2D`:

- **Sobel X** detects intensity changes from left to right, highlighting vertical edges.
- **Sobel Y** detects intensity changes from top to bottom, highlighting horizontal edges.

The two directional gradients were combined using the gradient magnitude:

`sqrt(Gx² + Gy²)`

## Results and observations

- The vertical-edge result outlines features with strong left-to-right changes, such as the sides of the cat, ears, and facial details.
- The horizontal-edge result highlights top-to-bottom changes, including the floor texture and horizontal boundaries around the body.
- The combined edge map shows both directions together and preserves the relative strength of each edge.
- A binary edge map is created by thresholding the combined map. It keeps only strong edges, which reduces weak texture details but may also remove subtle features.

## Threshold experiment

Different threshold values can be tested in the notebook:

- A lower threshold retains more details but can also preserve noise and floor texture.
- The default threshold of `60` gives a balanced result for this image.
- A higher threshold produces a cleaner image but removes weaker edges, such as fur details.

## Conclusion

Sobel kernels are effective for basic edge detection because they isolate horizontal and vertical brightness changes. Combining both responses produces a complete edge representation, while thresholding provides a cleaner binary edge map when only prominent boundaries are needed.
