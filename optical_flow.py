"""Optical flow computation using Lucas-Kanade algorithm."""
import cv2
import numpy as np


def build_pyramid(img: np.ndarray, levels: int) -> list:
    """Build Gaussian pyramid for an image."""
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def lucas_kanade_single_level(
    gframe1: np.ndarray,
    gframe2: np.ndarray,
    pts: np.ndarray,
    window_size: int,
    max_iterations: int,
    epsilon: float,
    initial_flows: np.ndarray = None,
) -> np.ndarray:
    """Lucas-Kanade optical flow at a single pyramid level."""
    h, w = gframe1.shape

    Ix = cv2.Sobel(gframe1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gframe1, cv2.CV_64F, 0, 1, ksize=3)

    half = window_size // 2
    flows = np.zeros((pts.shape[0], 2), dtype=np.float32)
    if initial_flows is not None:
        flows[:] = initial_flows

    for i, (x, y) in enumerate(pts):
        xi = int(round(x))
        yi = int(round(y))

        x0 = max(0, xi - half)
        x1 = min(w, xi + half + 1)
        y0 = max(0, yi - half)
        y1 = min(h, yi + half + 1)

        Ix_patch = Ix[y0:y1, x0:x1].astype(np.float64)
        Iy_patch = Iy[y0:y1, x0:x1].astype(np.float64)
        g1_patch = gframe1[y0:y1, x0:x1].astype(np.float64)

        Ix_flat = Ix_patch.ravel()
        Iy_flat = Iy_patch.ravel()

        Ixx = np.dot(Ix_flat, Ix_flat)
        Ixy = np.dot(Ix_flat, Iy_flat)
        Iyy = np.dot(Iy_flat, Iy_flat)

        A = np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=np.float64)

        det = Ixx * Iyy - Ixy * Ixy
        if det < 1e-6:
            flows[i] = np.nan
            continue

        A_inv = np.linalg.inv(A)

        u, v = flows[i]
        for _ in range(max_iterations):
            cx = xi + u
            cy = yi + v

            if cx < half or cx >= w - half or cy < half or cy >= h - half:
                break

            patch_h = y1 - y0
            patch_w = x1 - x0

            g2_patch = cv2.getRectSubPix(
                gframe2, (patch_w, patch_h), (cx, cy)
            ).astype(np.float64)

            diff_patch = g2_patch - g1_patch

            b = np.array(
                [
                    np.dot(Ix_flat, diff_patch.ravel()),
                    np.dot(Iy_flat, diff_patch.ravel()),
                ],
                dtype=np.float64,
            )

            du, dv = A_inv @ b
            u -= du
            v -= dv

            if abs(du) < epsilon and abs(dv) < epsilon:
                break

        flows[i] = (u, v)

    return flows


def find_alignment_between_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    points: np.ndarray | list = None,
    window_size: int = 15,
    max_iterations: int = 30,
    epsilon: float = 0.0001,
    pyramid_levels: int = 5,
) -> tuple | np.ndarray:
    """Find optical flow between two frames using pyramidal Lucas-Kanade."""
    gframe1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gframe2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if points is not None:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("`points` must be an array-like of shape (N, 2)")

        pyramid1 = build_pyramid(gframe1, pyramid_levels)
        pyramid2 = build_pyramid(gframe2, pyramid_levels)

        flows = np.zeros((pts.shape[0], 2), dtype=np.float32)

        for level in range(pyramid_levels - 1, -1, -1):
            scale = 2 ** level
            scaled_pts = pts / scale
            if level < pyramid_levels - 1:
                flows *= 2.0
            flows = lucas_kanade_single_level(
                pyramid1[level],
                pyramid2[level],
                scaled_pts,
                window_size,
                max_iterations,
                epsilon,
                initial_flows=flows,
            )

        return flows
