import cv2

# Read from video file instead of webcam (0)
cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("Cannot open video file")
    exit()

# Get video dimensions and FPS for saving output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

# Create VideoWriter to save the motion mask (isColor=False for grayscale mask)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("motion_output.mp4", fourcc, fps, (width, height), isColor=False)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Save the motion mask frame to output video
    out.write(fgmask)
    frame_count += 1

cap.release()
out.release()
print(f"Processed {frame_count} frames. Motion mask saved to motion_output.mp4")

