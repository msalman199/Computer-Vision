import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 2: Load and Visualize Thermal Images ---
image_path = "images/thermal1.jpg"  # Replace with your actual image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Image not found at '{image_path}'.")
    exit()

# Plot thermal colormap and save a copy for headless environments
plt.figure(figsize=(8, 6))
plt.imshow(img, cmap="inferno")
plt.title("Thermal Image")
plt.colorbar(label="Temperature Intensity")
plt.savefig("thermal_colormap.png", bbox_inches="tight")
plt.show()

# --- Step 3: Detect Heat Anomalies (Hotspots) ---
# Threshold to find hot zones (0-255 scale)
threshold = 200
_, hot_mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

# Find contours (hot regions)
contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save annotated hotspot image
cv2.imwrite("detected_hotspots.jpg", img_color)

# --- Step 4: Predict Failure Based on Heat Intensity ---
print(f"Total hotspots detected: {len(contours)}")

if len(contours) >= 2:
    print("⚠️  Warning: Multiple hotspots detected. Possible overheating risk.")
elif len(contours) == 1:
    print("⚠️  One hotspot detected. Monitor the component.")
else:
    print("✅  No critical heat anomalies detected.")

# Display the result window
cv2.imshow("Detected Hotspots", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

