import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found. Please ensure 'sample_image.jpg' is in the directory.")
        return

   
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

  
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')

    axes[1].imshow(np.abs(grad_x), cmap='gray')
    axes[1].set_title('Vertical Edges (Sobel X)')
    axes[1].axis('off')

    axes[2].imshow(gradient_magnitude, cmap='gray')
    axes[2].set_title('Combined Edge Magnitude')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    detect_edges('sample_image.jpg')
