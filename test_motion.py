import cv2
import numpy as np
import os

os.makedirs("debug_outputs", exist_ok=True)

h, w = 480, 640

frame1 = np.ones((h, w, 3), dtype=np.uint8) * 255
cv2.rectangle(frame1, (100, 150), (200, 250), (0, 0, 255), -1)
cv2.imwrite("debug_outputs/square_frame_0.png", frame1)
print("Saved square_frame_0.png (baseline)")

frame2 = np.ones((h, w, 3), dtype=np.uint8) * 255
cv2.rectangle(frame2, (110, 153), (210, 253), (0, 0, 255), -1)
cv2.imwrite("debug_outputs/square_frame_1.png", frame2)
print("Saved square_frame_1.png (translation only)")

frame3 = np.ones((h, w, 3), dtype=np.uint8) * 255
cv2.rectangle(frame3, (120, 156), (220, 256), (0, 0, 255), -1)
center = (w // 2, h // 2)
rotation_angle = 10
M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
frame3 = cv2.warpAffine(frame3, M, (w, h), borderValue=(255, 255, 255))
cv2.imwrite("debug_outputs/square_frame_2.png", frame3)
print("Saved square_frame_2.png (translation + 3° rotation)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("debug_outputs/square_motion.mp4", fourcc, 2.0, (w, h))

out.write(frame1)
out.write(frame2)
out.write(frame3)

out.release()
print("Saved square_motion.mp4 with 3 frames")
