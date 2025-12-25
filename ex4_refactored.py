"""Main video stabilization script."""
import cv2

from video_io import read_video_to_frames, save_stabilized_video
from optical_flow import find_alignment_between_frames
from transform import compute_transform_matrix
from stabilization import stabilize_y
from debug_utils import (
    debug_print_flow,
    debug_visualize_flows,
    apply_and_visualize_transforms,
)


def compute_transform_matrices(frames, output_dir, max_corners=200, quality_level=0.001, min_distance=30):
    """Compute transformation matrices for all consecutive frame pairs."""
    transform_matrices = []
    print(f"Processing {len(frames)} frames...")

    for idx in range(len(frames) - 1):
        print(f"Frame {idx}/{len(frames)-1}", end="\r")
        g = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            g, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance
        )
        points = corners.reshape(-1, 2)
        flows = find_alignment_between_frames(frames[idx], frames[idx + 1], points)

        M, theta, tx, ty = compute_transform_matrix(points, flows)
        transform_matrices.append(M)

        if idx < 10:
            debug_print_flow(idx, theta, tx, ty, points, flows)
            debug_visualize_flows(
                frames[idx], points, flows, f"{output_dir}/flow_frame_{idx}.png"
            )

    print("\nDone processing frames.")
    return transform_matrices


def main():
    frames, fps, video_name, output_dir = read_video_to_frames()
    transform_matrices = compute_transform_matrices(frames, output_dir)
    apply_and_visualize_transforms(frames, transform_matrices, output_dir, num_frames=10)

    stabilized_y_rot_frames = stabilize_y(frames, transform_matrices)
    output_path_y_rot = f"{output_dir}/stabilized.mp4"
    save_stabilized_video(stabilized_y_rot_frames, output_path_y_rot, fps=fps)
    print(f"Y-rotation stabilized video saved to {output_path_y_rot}")
    print(
        f"Output video: {len(stabilized_y_rot_frames)} frames at {fps} FPS ({len(stabilized_y_rot_frames)/fps:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
