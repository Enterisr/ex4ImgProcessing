"""Video stabilization functions."""
import cv2
import numpy as np


def moving_average(curve, radius):
    """Smooth a curve using moving average filter."""
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), "edge")
    curve_smoothed = np.convolve(curve_pad, f, mode="same")
    return curve_smoothed[radius:-radius]


def stabilize_y(frames, transform_matrices, smoothing_radius=20, crop_ratio=1.10):
    """
    Stabilize video by removing Y-axis translation and rotation,
    while preserving X-axis movement from the original video.
    Uses trajectory smoothing approach.
    """
    stabilized_frames = []
    h, w = frames[0].shape[:2]

    # Extract transforms from matrices
    transforms = []
    for M in transform_matrices:
        tx = M[0, 2]
        ty = M[1, 2]
        theta = np.arctan2(M[1, 0], M[0, 0])
        transforms.append([tx, ty, theta])
    transforms = np.array(transforms)

    # Compute trajectory
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory
    smoothed = np.copy(trajectory)
    for transform_idx in range(3):
        smoothed[:, transform_idx] = moving_average(
            trajectory[:, transform_idx], radius=smoothing_radius
        )

    # Calculate stabilization transforms
    difference = smoothed - trajectory

    # Apply stabilization (Y and rotation only, preserve X)
    for idx in range(len(frames) - 1):
        dx = 0  # Preserve X movement
        dy = difference[idx, 1]  # Stabilize Y
        da = difference[idx, 2]  # Stabilize rotation

        m = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(da), crop_ratio)
        m[0, 2] += dx
        m[1, 2] += dy

        frame_stabilized = cv2.warpAffine(frames[idx], m, (w, h))
        stabilized_frames.append(frame_stabilized)

    stabilized_frames.append(frames[-1])
    return stabilized_frames
