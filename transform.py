"""Transform estimation and computation functions."""
import cv2
import numpy as np
from optical_flow import build_pyramid


def compute_transform_matrix(points, flows):
    """Compute transformation matrix using OpenCV's estimateAffinePartial2D with RANSAC."""
    # Filter out invalid flows
    valid_mask = ~np.isnan(flows).any(axis=1)
    valid_points = points[valid_mask]
    valid_flows = flows[valid_mask]

    src_pts = valid_points.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = (valid_points + valid_flows).reshape(-1, 1, 2).astype(np.float32)

    rot_matrix, inliers = cv2.estimateAffine2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=100,
        confidence=0.99,
    )
    
    # Force no scale by reconstructing with unit scale
    if rot_matrix is not None:
        theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
        tx = rot_matrix[0, 2]
        ty = rot_matrix[1, 2]
        
        # Reconstruct rotation matrix with no scale
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_matrix = np.array([
            [cos_theta, -sin_theta, tx],
            [sin_theta, cos_theta, ty]
        ], dtype=np.float64)


    # Extract parameters for debugging
    theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    tx = rot_matrix[0, 2]
    ty = rot_matrix[1, 2]

    M = np.vstack([rot_matrix, [0, 0, 1]])
    return M, theta, tx, ty


def estimate_global_transform_lk(
    frame1,
    frame2,
    window_size: int = 15,
    pyramid_levels: int = 5,
    max_iterations: int = 20,
    epsilon: float = 1e-4,
    grad_threshold: float = 1.0,
):
    """Estimate global rotation (alpha) and translation (dx,dy) using the linearized LK model:
       u(x,y)=dx - alpha*y, v(x,y)=dy + alpha*x (x,y measured around the image center).
       Returns a 3x3 affine matrix M and parameters (theta, tx, ty)."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32)

    pyr1 = build_pyramid(g1, pyramid_levels)
    pyr2 = build_pyramid(g2, pyramid_levels)

    alpha = 0.0
    dx = 0.0
    dy = 0.0

    for level in range(pyramid_levels - 1, -1, -1):
        I1 = pyr1[level]
        I2 = pyr2[level]

        h, w = I1.shape
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        x_c = xx - cx
        y_c = yy - cy

        Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3)

        half = window_size // 2
        grad_mag = np.sqrt(Ix * Ix + Iy * Iy)
        mask = (
            (xx >= half)
            & (xx < w - half)
            & (yy >= half)
            & (yy < h - half)
            & (grad_mag > grad_threshold)
        )

        if np.count_nonzero(mask) < 10:
            continue

        Ix_m = Ix[mask].reshape(-1)
        Iy_m = Iy[mask].reshape(-1)
        x_c_m = x_c[mask].reshape(-1)
        y_c_m = y_c[mask].reshape(-1)

        for _ in range(max_iterations):
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            M_aff = np.array(
                [[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float32
            )
            I2_w = cv2.warpAffine(
                I2, M_aff, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

            It = (I2_w - I1)[mask].reshape(-1)

            # A * [dx, dy, alpha]^T = -It  with A = [Ix, Iy, (-Ix*y + Iy*x)]
            A0 = Ix_m
            A1 = Iy_m
            A2 = (-Ix_m * y_c_m + Iy_m * x_c_m)
            A = np.stack([A0, A1, A2], axis=1)

            ATA = A.T @ A
            ATb = A.T @ (-It)
            ATA += np.eye(3, dtype=np.float32) * 1e-6  # damping

            delta = np.linalg.solve(ATA, ATb).astype(np.float32)
            ddx, ddy, dalpha = float(delta[0]), float(delta[1]), float(delta[2])

            dx += ddx
            dy += ddy
            alpha += dalpha

            if max(abs(ddx), abs(ddy), abs(dalpha)) < epsilon:
                break

        if level > 0:
            dx *= 2.0
            dy *= 2.0  # alpha stays unchanged

    # Build final transformation matrix that rotates around image center
    # The linearized model assumes rotation around center, so dx,dy are already 
    # translations at the original scale
    h_orig, w_orig = g1.shape
    cx_orig = (w_orig - 1) / 2.0
    cy_orig = (h_orig - 1) / 2.0
    
    # Create rotation matrix around image center
    # M = T(cx,cy) * R(alpha) * T(-cx,-cy) + translation
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    
    # Rotation around center then translate
    rot_matrix = np.array([
        [cos_a, -sin_a, -cx_orig * cos_a + cy_orig * sin_a + cx_orig + dx],
        [sin_a, cos_a, -cx_orig * sin_a - cy_orig * cos_a + cy_orig + dy]
    ], dtype=np.float64)

    M = np.vstack([rot_matrix, [0, 0, 1]])
    theta = float(alpha)
    tx = float(dx)
    ty = float(dy)
    return M, theta, tx, ty
