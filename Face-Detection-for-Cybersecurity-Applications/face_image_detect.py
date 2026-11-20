import cv2

# Load Haar cascade from the current directory
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error: Could not load 'haarcascade_frontalface_default.xml'.")
    exit()

# Load image
image = cv2.imread("face.jpg")
if image is None:
    print("Error: Could not load 'face.jpg'. Make sure it is in the same folder.")
    exit()

# Convert to grayscale and detect faces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"Faces detected: {len(faces)}")

# Draw green rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the output image (instead of cv2.imshow for headless EC2)
cv2.imwrite("face_detected.jpg", image)
print("Saved output image as face_detected.jpg")

