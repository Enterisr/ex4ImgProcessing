import cv2
import numpy as np
import imageio.v3 as iio


# ============================================================================
# VIDEO I/O FUNCTIONS
# ============================================================================

def read_video_to_frames():
    input_path = "input/boat.mp4"
    #input_path = "debug_outputs/square_motion.mp4"
    
    # Read video with metadata
    vid = iio.imread(input_path, plugin="pyav")
    
    # Get FPS from video
    props = iio.improps(input_path, plugin="pyav")
    fps = props.fps if hasattr(props, 'fps') else 30
    
    print(f"Input video: {len(vid)} frames at {fps} FPS ({len(vid)/fps:.2f} seconds)")
    
    return vid, fps


def save_stabilized_video(frames, output_path, fps=30):
    iio.imwrite(
        output_path, np.array(frames), fps=fps, plugin="pyav", codec="libx264"
    )


# ============================================================================
# OPTICAL FLOW FUNCTIONS
# ============================================================================

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
    window_size: int = 25,
    max_iterations: int =100,
    epsilon: float = 0.0001,
    pyramid_levels: int = 5,
) -> tuple | np.ndarray:

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


# ============================================================================
# TRANSFORM COMPUTATION FUNCTIONS
# ============================================================================

def find_rotation(curr_points, next_points):
    p1, p2 = curr_points
    a = p2 - p1
    b = next_points[1] - next_points[0]
    cross = a[0] * b[1] - a[1] * b[0]  
    dot = a @ b
    return np.arctan2(cross, dot)



def compute_transform_matrix(points, flows):
    """Compute transformation matrix using OpenCV's estimateAffinePartial2D with RANSAC."""
    # Filter out invalid flows
    valid_mask = ~np.isnan(flows).any(axis=1)
    valid_points = points[valid_mask]
    valid_flows = flows[valid_mask]
    
    if len(valid_points) < 3:
        # Not enough points, return identity transformation
        tx = np.median(valid_flows[:, 0]) if len(valid_flows) > 0 else 0
        ty = np.median(valid_flows[:, 1]) if len(valid_flows) > 0 else 0
        theta = 0
        center_x, center_y = points[0] if len(points) > 0 else (0, 0)
        
        rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(theta), 1.0)
        rot_matrix[0, 2] += tx
        rot_matrix[1, 2] += ty
    else:
        # Use OpenCV's estimateAffinePartial2D with RANSAC
        src_pts = valid_points.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = (valid_points + valid_flows).reshape(-1, 1, 2).astype(np.float32)
        
        # estimateAffinePartial2D estimates similarity transform (rotation + uniform scale + translation)
        # with RANSAC for robustness
        rot_matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, 
            dst_pts, 
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=100,
            confidence=0.99
        )
        
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


def compute_transform_matrices(frames):
    transform_matrices = []
    print(f"Processing {len(frames)} frames...")
    
    for idx in range(len(frames) - 1):
        print(f"Frame {idx}/{len(frames)-1}", end="\r")
        g = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            g, maxCorners=50, qualityLevel=0.001, minDistance=60
        )
        points = corners.reshape(-1, 2)
        flows = find_alignment_between_frames(frames[idx], frames[idx + 1], points)
        
        M, theta, tx, ty = compute_transform_matrix(points, flows)
        transform_matrices.append(M)
        
        if idx < 10:
            debug_print_flow(idx, theta, tx, ty, points, flows)
            debug_visualize_flows(
                frames[idx], points, flows, f"debug_outputs/flow_frame_{idx}.png"
            )
    
    print("\nDone processing frames.")
    return transform_matrices


# ============================================================================
# STABILIZATION FUNCTIONS
# ============================================================================

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

def stabilize_y(frames, transform_matrices, smoothing_radius=50, crop_ratio=1.10):
    stabilized_frames = []
    h, w = frames[0].shape[:2]

    transforms = []
    for M in transform_matrices:
        tx = M[0, 2]
        ty = M[1, 2]
        theta = np.arctan2(M[1, 0], M[0, 0])
        transforms.append([tx, ty, theta])
    transforms = np.array(transforms)

    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=smoothing_radius)

    difference = smoothed_trajectory - trajectory
    
    for idx in range(len(frames) - 1):
        dx = difference[idx, 0]
        dy = difference[idx, 1]
        da = difference[idx, 2]

        m = cv2.getRotationMatrix2D((w/2, h/2), np.degrees(da), crop_ratio)
        m[0, 2] += dx
        m[1, 2] += dy

        frame_stabilized = cv2.warpAffine(frames[idx], m, (w, h))
        stabilized_frames.append(frame_stabilized)

    stabilized_frames.append(frames[-1])
    return stabilized_frames
# ============================================================================
# DEBUG AND VISUALIZATION FUNCTIONS
# ============================================================================

def debug_print_flow(idx, theta, tx, ty, points, flows):
    """Print flow debugging information."""
    print(f"\nFrame {idx}: theta={np.degrees(theta):.2f}°, tx={tx:.2f}, ty={ty:.2f}")
    print(f"  Point 1: {points[0]} -> {points[0] + flows[0]}, flow={flows[0]}")
    print(f"  Point 2: {points[1]} -> {points[1] + flows[1]}, flow={flows[1]}")


def debug_visualize_flows(frame, points, flows, output_path):
    """Visualize flow vectors on frame and save to file."""
    debug_frame = frame.copy()
    for pt, flow in zip(points, flows):
        if not np.isnan(flow).any():
            pt_int = tuple(pt.astype(int))
            end_pt = tuple((pt + flow).astype(int))
            cv2.arrowedLine(debug_frame, pt_int, end_pt, (0, 255, 0), 2)
    debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, debug_frame)


def apply_and_visualize_transforms(frames, transform_matrices, num_frames=4):
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
        overlay_path = f"debug_outputs/overlay_frame_{i}_on_{i+1}.png"
        cv2.imwrite(
            overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        print(f"  Saved {overlay_path}")

    print("\nTransform matrices applied and saved to debug_outputs/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    frames, fps = read_video_to_frames()
    transform_matrices = compute_transform_matrices(frames)
    apply_and_visualize_transforms(frames, transform_matrices, num_frames=10)
    
    # Y and rotation only stabilization (preserves X movement)
    stabilized_y_rot_frames = stabilize_y(frames, transform_matrices)
    output_path_y_rot = "debug_outputs/stabilized_video_y_rotation_only.mp4"
    save_stabilized_video(stabilized_y_rot_frames, output_path_y_rot, fps=fps)
    print(f"Y-rotation stabilized video saved to {output_path_y_rot}")
    print(f"Output video: {len(stabilized_y_rot_frames)} frames at {fps} FPS ({len(stabilized_y_rot_frames)/fps:.2f} seconds)")


if __name__ == "__main__":
    main()
