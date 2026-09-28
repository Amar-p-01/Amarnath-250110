import cv2
import matplotlib.pyplot as plt
import numpy as np

Sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]], dtype=np.float32)

Sobel_y = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]], dtype=np.float32)

Scharr_x = np.array([[-3,  0,  3], 
                     [-10, 0, 10], 
                     [-3,  0,  3]], dtype=np.float32)

Scharr_y = np.array([[-3, -10, -3], 
                     [ 0,   0,  0], 
                     [ 3,  10,  3]], dtype=np.float32)

blur = np.ones((3, 3), dtype=np.float32) / 9.0

gaussian_blur = np.array([
    [1, 2, 1], [2, 4, 2], [1, 2, 1]
], dtype=np.float32) / 16

path = "/ghost.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# applying gaussian blur
img_blur = cv2.filter2D(img, -1, gaussian_blur)

# applying sobel
g_x = cv2.filter2D(img_blur, cv2.CV_64F, Scharr_x)
g_y = cv2.filter2D(img_blur, cv2.CV_64F, Scharr_y)

mag = np.sqrt(g_x**2 + g_y**2)
ang = np.arctan2(g_y, g_x) * 180 / np.pi

# Non Max suppression
def non_max_suppression(mag, ang):
    H, W = mag.shape
    out = np.zeros_like(mag)
    angle = ang % 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = mag[i, j-1], mag[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = mag[i-1, j+1], mag[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = mag[i-1, j], mag[i+1, j]
            else:
                n1, n2 = mag[i-1, j-1], mag[i+1, j+1]

            if mag[i, j] >= n1 and mag[i, j] >= n2:
                out[i, j] = mag[i, j]
    return out

mag = non_max_suppression(mag, ang)

low = np.percentile(mag[mag > 0], 60) 
high = np.percentile(mag[mag > 0], 90)

edges = np.zeros_like(mag, dtype=np.uint8)
edges[mag >= high] = 255
edges[mag < low] = 0

plt.imshow(edges, cmap = 'grey')
plt.axis('off')
plt.show()