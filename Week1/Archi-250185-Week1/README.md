## Experimentation and Results Analysis

### 1. First-Order Gradients (Sobel & Scharr)
* **Directional Sensitivity:** Horizontal kernels ($X$) successfully isolate vertical features (e.g., the vertical stems of the letters in "BRAIN"), while vertical kernels ($Y$) isolate horizontal lines (e.g., baseline underlines). Combining them via Euclidean magnitude ($\sqrt{G_x^2 + G_y^2}$) unifies these boundaries into a complete outline.
* **Sobel vs. Scharr:** The Scharr operator produced bolder, more continuous edge outlines. Because Scharr employs higher central weights ($\pm 10$ vs. $\pm 2$ for Sobel), it increases gradient sensitivity and improves rotational symmetry along curved paths like the brain sulci.

### 2. Second-Order Derivatives (Laplacian)
* **Single-Pass Efficiency:** Unlike first-order filters that require computing $X$ and $Y$ gradients separately before combining them, the Laplacian operator isolates sharp boundaries in a single convolution pass by measuring zero-crossings.
* **Line Thinning:** The Laplacian outputs produced significantly thinner, more precise single-pixel-width boundary lines around the text and illustration. The **Diagonal Laplacian** variant further improved angular and curved continuity by incorporating diagonal neighborhood weights.

### 3. Custom Multi-Stage Pipeline (Gaussian + Gradients + Thresholding)
* **Noise Reduction:** Adding a Gaussian pre-filtering layer successfully eliminated background grain.
* **Comparative Insight:** While the pipeline effectively isolated strong structural features, the resulting text outlines appeared thicker and slightly more prone to fragmentation compared to the clean, razor-thin lines produced by the Laplacian operator. This highlights how second-order derivatives naturally snap to precise zero-crossings on vector-style graphics.


