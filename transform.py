"""Transform estimation and computation functions."""
import cv2
import numpy as np


def compute_transform_matrix(points, flows):
    """Compute transformation matrix using OpenCV's estimateAffinePartial2D with RANSAC."""
    # Filter out invalid flows
    valid_mask = ~np.isnan(flows).any(axis=1)
    valid_points = points[valid_mask]
    valid_flows = flows[valid_mask]

    src_pts = valid_points.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = (valid_points + valid_flows).reshape(-1, 1, 2).astype(np.float32)

    # estimateAffine2D estimates full affine transform (rotation + translation + shear + scale)
    # Use fullAffine=False to restrict to similarity transform but we'll compute rigid transform manually
    # estimateRigidTransform would be ideal but it's deprecated, so we use estimateAffine2D
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

    if rot_matrix is None:
        # Fallback if estimation fails
        tx = np.median(valid_flows[:, 0])
        ty = np.median(valid_flows[:, 1])
        theta = 0
        h, w = 480, 640  # default frame size
        center_x, center_y = w / 2, h / 2

        rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.0)
        rot_matrix[0, 2] = tx
        rot_matrix[1, 2] = ty

    # Extract parameters for debugging
    theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    tx = rot_matrix[0, 2]
    ty = rot_matrix[1, 2]

    M = np.vstack([rot_matrix, [0, 0, 1]])
    return M, theta, tx, ty
