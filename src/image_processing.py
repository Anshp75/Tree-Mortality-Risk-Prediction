import cv2
import numpy as np
from PIL import Image

def analyze_tree(image):
    # --- 1. IMAGE RESTORATION (Denoising & Normalization) ---
    # Converts PIL to OpenCV and removes noise/filters from edited photos
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Fast Non-Local Means Denoising for high-quality restoration
    img_restored_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    
    # Convert restored image back to PIL for the UI display
    img_restored_pil = Image.fromarray(cv2.cvtColor(img_restored_cv, cv2.COLOR_BGR2RGB))
    
    # Convert to HSV for accurate color-space diagnostics
    hsv = cv2.cvtColor(img_restored_cv, cv2.COLOR_BGR2HSV)
    
    # --- 2. MULTI-COLOR SEGMENTATION (Health Check) ---
    # Healthy Green, Thirsty Yellow/Pale, and Dead Brown/Grey
    mask_g = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_y = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([34, 255, 255]))
    mask_b = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([19, 255, 200]))
    
    g_pixels = np.sum(mask_g > 0)
    y_pixels = np.sum(mask_y > 0)
    b_pixels = np.sum(mask_b > 0)
    total_organic = g_pixels + y_pixels + b_pixels + 1
    
    # --- 3. SUNLIGHT & LUMINANCE ANALYSIS ---
    # Analyze the 'Value' (Brightness) channel of the leaves to check for shading
    leaf_val = cv2.bitwise_and(hsv[:,:,2], hsv[:,:,2], mask=mask_g)
    avg_brightness = np.sum(leaf_val) / (g_pixels + 1)
    
    # --- 4. CALCULATIONS (Binary & Regression) ---
    # Regression: Weighted Hydration Score (Green=1.0, Yellow=0.4, Brown=0.0)
    hydration = int(((g_pixels * 1.0) + (y_pixels * 0.4)) / total_organic * 100)
    
    # Binary Classification: Life Status
    pixel_area = img_array.shape[0] * img_array.shape[1]
    if (g_pixels + y_pixels) / pixel_area < 0.03:
        status = "DEAD / STUMP"
        hydration = 0 # Dead trees have no hydration
    else:
        status = "LIVE"
        
    return status, hydration, avg_brightness, img_restored_pil