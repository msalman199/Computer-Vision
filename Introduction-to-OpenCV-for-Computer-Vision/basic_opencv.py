import cv2

# --- Step 1: Load and Display an Image ---
# Make sure sample.jpg is in the same folder
image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found. Make sure 'sample.jpg' is in the same directory.")
    exit()

cv2.imshow("Original Image", image)

# --- Step 2: Save the Image ---
cv2.imwrite("output.jpg", image)
print("Image saved as output.jpg")

# --- Step 3: Color Conversion and Transformations ---
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.imwrite("grayscale.jpg", gray)
print("Grayscale image saved as grayscale.jpg")

# Resize the image to 200x200 pixels
resized = cv2.resize(image, (200, 200))
cv2.imshow("Resized Image", resized)
cv2.imwrite("resized.jpg", resized)
print("Resized image saved as resized.jpg")

# Wait for any key press to close all image windows
print("Press any key while focused on an image window to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()
