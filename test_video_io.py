import cv2
import sys

video_path = "input.mp4"   # 🔹 make sure this file exists here!

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: could not open video file: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video props -> FPS:", fps, "Width:", w, "Height:", h)

if w == 0 or h == 0:
    print("Width/Height is 0. Bad input video or codec issue.")
    sys.exit(1)

# use AVI + XVID (very safe on Windows)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("test_output.avi", fourcc, fps if fps > 0 else 25.0, (w, h))

if not out.isOpened():
    print("Error: could not open VideoWriter")
    sys.exit(1)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames. Breaking.")
        break

    frame_idx += 1
    if frame_idx % 30 == 0:
        print("Processed frame:", frame_idx)

    out.write(frame)

cap.release()
out.release()
print("Done. Saved test_output.avi")
