import cv2
import numpy as np
import pathlib


def create_motion_composition(frames, transform_matrices, output_path, stride=1):
    """Feathered mosaic using per-frame soft masks to avoid seams/stripes.

    Stride lets you thin frames; feathering uses distance transform on the warped
    mask so overlaps blend smoothly instead of hard seams.
    """

    output_path = pathlib.Path(output_path) / "motion_composited.png"
    h, w = frames[0].shape[:2]

    cumulative_transforms = [np.eye(3)]
    current_T = np.eye(3)
    for M in transform_matrices:
        current_T = current_T @ M
        cumulative_transforms.append(current_T)

    base_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(
        -1, 1, 2
    )

    all_corners = []
    for T in cumulative_transforms:
        transformed_corners = cv2.transform(base_corners, T[:2])
        all_corners.append(transformed_corners)

    all_corners = np.vstack(all_corners)
    x_min, y_min = all_corners[:, 0, :].min(axis=0)
    x_max, y_max = all_corners[:, 0, :].max(axis=0)

    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))

    T_offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    print(f"Canvas size: {canvas_w}x{canvas_h}, Offset: ({-x_min:.1f}, {-y_min:.1f})")

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i, frame in enumerate(frames):
        if i % stride != 0:
            continue
        if i >= len(cumulative_transforms):
            break

        T_global = cumulative_transforms[i]
        T_final = T_offset @ T_global
        warped_frame = cv2.warpAffine(
            frame,
            T_final[:2],
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask = np.any(warped_frame > 0, axis=2).astype(np.uint8)
        if not np.any(mask):
            continue

        # Soft weights via distance transform to feather edges.
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist = dist / (dist.max() + 1e-6)
        dist = dist.astype(np.float32)

        canvas += warped_frame.astype(np.float32) * dist[:, :, None]
        weight_map += dist

    # Normalize by weights to avoid bright/white accumulation
    nonzero = weight_map > 0
    canvas_norm = np.zeros_like(canvas, dtype=np.float32)
    canvas_norm[nonzero] = canvas[nonzero] / weight_map[nonzero, None]

    canvas_uint8 = np.clip(canvas_norm, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, canvas_uint8)
    print(f"Saved composition to {output_path}")