"""Debug and visualization utilities."""
import cv2
import numpy as np


def debug_print_flow(idx, theta, tx, ty, points, flows):
    """Print flow debugging information."""
    print(f"\nFrame {idx}: theta={np.degrees(theta):.2f}°, tx={tx:.2f}, ty={ty:.2f}")


def debug_visualize_flows(frame, points, flows, output_path):
    """Visualize flow vectors on frame and save to file."""
    debug_frame = frame.copy()
    if points is not None and flows is not None:
        for pt, flow in zip(points, flows):
            if not np.isnan(flow).any():
                pt_int = tuple(pt.astype(int))
                end_pt = tuple((pt + flow).astype(int))
                cv2.arrowedLine(debug_frame, pt_int, end_pt, (0, 255, 0), 2)
    debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, debug_frame)


def apply_and_visualize_transforms(frames, transform_matrices, output_dir, num_frames=4):
    """Apply transforms to first N frames and create overlay visualizations."""
    print("\nApplying first {} transform matrices...".format(num_frames))
    for i in range(min(num_frames, len(transform_matrices))):
        print(f"\nApplying transform {i}...")
        frame = frames[i]
        next_frame = frames[i + 1]
        M = transform_matrices[i]
        h, w = frame.shape[:2]

        print(f"  Transform matrix M[{i}]:")
        print(f"    {M[0]}")
        print(f"    {M[1]}")
        print(f"    {M[2]}")

        theta = np.arctan2(M[1, 0], M[0, 0])
        tx, ty = M[0, 2], M[1, 2]
        print(f"  Extracted: theta={np.degrees(theta):.4f}°, tx={tx:.4f}, ty={ty:.4f}")

        transformed = cv2.warpAffine(frame, M[:2, :], (w, h))

        overlay = cv2.addWeighted(
            next_frame.astype(np.float32), 0.5, transformed.astype(np.float32), 0.5, 0
        )
        overlay_path = f"{output_dir}/overlay_frame_{i}_on_{i+1}.png"
        cv2.imwrite(
            overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        print(f"  Saved {overlay_path}")

    print(f"\nTransform matrices applied and saved to {output_dir}")
