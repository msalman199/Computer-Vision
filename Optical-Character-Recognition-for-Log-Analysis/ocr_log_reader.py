import os
import cv2
import pytesseract

# 1. Set and verify image path
img_path = "images/log1.png"  # Replace with your image path
if not os.path.exists(img_path):
    print(f"Error: Image not found at '{img_path}'")
    exit()

# 2. Load image and convert to grayscale
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Preprocess for better OCR accuracy
# Uncomment the line below if your log image has white text on a dark background:
# gray = cv2.bitwise_not(gray)

blur = cv2.medianBlur(gray, 3)
thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]

# 4. Extract text using Tesseract
text = pytesseract.image_to_string(thresh)

print("=== Extracted Log Content ===")
print(text)

# 5. Analyze extracted text for common log alerts
keywords = ["ERROR", "FAILED", "WARNING", "CRITICAL"]

print("\n=== Alerts Found ===")
for line in text.split('\n'):
    for keyword in keywords:
        if keyword in line.upper():
            print(f"[{keyword}] {line.strip()}")

