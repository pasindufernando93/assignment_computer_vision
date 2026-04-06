import cv2

# Load image 
img_bgr = cv2.imread("/a1images/highlights_and_shadows.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # for matplotlib display

print(f"Image shape : {img_rgb.shape}")
print(f"Dtype       : {img_rgb.dtype}")


