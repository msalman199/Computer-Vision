import time
import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# CONFIGURATION: Set to True to force CPU mode (Step 2)
FORCE_CPU = False
# ==========================================

if FORCE_CPU:
    tf.config.set_visible_devices([], 'GPU')

use_gpu = len(tf.config.get_visible_devices('GPU')) > 0
device_label = "GPU" if use_gpu else "CPU"
print(f"Running on: {device_label}")

# Initialize webcam (0 is the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Pre-define the 3x3 box blur kernel outside the loop for efficiency
kernel = tf.constant([[1 / 9.0] * 3] * 3, dtype=tf.float32, shape=[3, 3, 1, 1])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for consistent processing and convert to grayscale
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Start timing
    start = time.time()

    # Run 2D convolution blur via TensorFlow
    img_tensor = tf.convert_to_tensor(gray, dtype=tf.float32)
    img_tensor = tf.reshape(img_tensor, [1, 480, 640, 1])
    blurred = tf.nn.conv2d(img_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
    blurred_np = blurred.numpy().squeeze()

    # End timing (convert to milliseconds)
    elapsed = (time.time() - start) * 1000

    # Overlay device and timing metrics on the original frame
    cv2.putText(
        frame,
        f"[{device_label}] Processing Time: {elapsed:.2f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # Display both video feeds
    cv2.imshow("Original", frame)
    cv2.imshow(f"Real-time Blurred ({device_label})", blurred_np.astype(np.uint8))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

