# Week 1 - Edge Detection

Used Sobel kernels to detect edges in an image — one kernel for vertical 
edges, one for horizontal, then combined both.

## How it works
1. Convert image to grayscale
2. Apply Sobel (dx=1,dy=0 for vertical edges, dx=0,dy=1 for horizontal)
3. Used CV_64F depth since gradients go negative, then converted back to 
   8-bit for display
4. Combined both with addWeighted
