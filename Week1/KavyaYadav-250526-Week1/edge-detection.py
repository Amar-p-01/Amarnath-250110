import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image_path, kernel_type='sobel'):
   
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found. Please check the image path.")

   
    if kernel_type == 'sobel':
        # Sobel Operator (Balances edge detection and noise smoothing)
        Kx = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])
        
        Ky = np.array([[-1, -2, -1], 
                       [ 0,  0,  0], 
                       [ 1,  2,  1]])
                       
    elif kernel_type == 'scharr':
        # Scharr Operator (More sensitive to fine details/gradients)
        Kx = np.array([[-3, 0, 3], 
                       [-10, 0, 10], 
                       [-3, 0, 3]])
        
        Ky = np.array([[-3, -10, -3], 
                       [ 0,  0,  0], 
                       [ 3,  10,  3]])
    else:
        raise ValueError("Unsupported kernel type. Choose 'sobel' or 'scharr'.")

    
    grad_x = cv2.filter2D(img, cv2.CV_64F, Kx)
    grad_y = cv2.filter2D(img, cv2.CV_64F, Ky)

   
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    

    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return img, abs_grad_x, abs_grad_y, magnitude

def plot_results(img, grad_x, grad_y, magnitude, title_prefix):

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(grad_x, cmap='gray')
    plt.title(f'{title_prefix} - Vertical Edges')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(grad_y, cmap='gray')
    plt.title(f'{title_prefix} - Horizontal Edges')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(magnitude, cmap='gray')
    plt.title(f'{title_prefix} - Magnitude')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_file = 'sample-edge-detection.jpg'  
    
    
    img, sx, sy, smag = detect_edges(image_file, kernel_type='sobel')
    plot_results(img, sx, sy, smag, "Sobel")
    

    img, cx, cy, cmag = detect_edges(image_file, kernel_type='scharr')
    plot_results(img, cx, cy, cmag, "Scharr")