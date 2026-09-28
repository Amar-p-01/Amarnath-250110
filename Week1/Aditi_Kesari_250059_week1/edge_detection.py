import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
def sobel_transform(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return None
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    edge_strength = np.sqrt(np.square(sobelx) + np.square(sobely))
    edge_strength = cv2.normalize(edge_strength, None, 0, 255, cv2.NORM_MINMAX)
    edge_strength = edge_strength.astype(np.uint8)
    return edge_strength

def canny_edge(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return None
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    canny_image = cv2.Canny(blurred_image, 100, 200)
    return canny_image

# 1. FIX THE PATHS: Ensure these point to actual image files (e.g., .jpg or .png)
image_file = "/home/aditi-kesari/IITM15.0-CV-Camp/Week1/Aditi_Kesari_250059_week1/task1/input_image.jpg"

sobel_1 = sobel_transform(image_file)
canny_1 = canny_edge(image_file)

# 2. SAFETY CHECK: Only plot if the images successfully loaded
if sobel_1 is not None and canny_1 is not None:
    plt.figure(figsize=(15,10))
    
    plt.subplot(1,2,1)
    plt.imshow(sobel_1, cmap='gray')
    plt.title("Sobel Transform")
    
    plt.subplot(1,2,2)
    plt.imshow(canny_1, cmap='gray')
    plt.title("Canny Edge Detection")
    
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    # 3. Save the file inside the newly confirmed directory
    out_filename = f"{out_dir}/sample_{idx+1}_result.png"
    plt.savefig(out_filename)
    plt.close()
    print(f"Saved: {out_filename}")
else:
    print("Plotting skipped because the image could not be loaded. Please check your file paths.")
