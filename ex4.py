import cv2
import numpy as np
import imageio.v3 as iio

def read_video_to_frames():
    #vid = iio.imread("input/boat.mp4", plugin="pyav")
    vid = iio.imread("debug_outputs/square_motion.mp4", plugin="pyav")
    return vid
    
def find_alignment_between_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    points: np.ndarray | list = None,
    window_size: int = 50,
    max_iterations: int =50,
    epsilon: float = 0.01,
) -> tuple | np.ndarray:

    gframe1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gframe2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gframe1.shape

    Ix = cv2.Sobel(gframe1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gframe1, cv2.CV_64F, 0, 1, ksize=3)

    if points is not None:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2: 
            raise ValueError("`points` must be an array-like of shape (N, 2)")

        half = window_size // 2
        flows = np.empty((pts.shape[0], 2), dtype=np.float32)
        flows[:] = np.nan

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

            A_inv = np.linalg.inv(A)

            u, v = 0.0, 0.0
            for _ in range(max_iterations):

                cx = xi + u
                cy = yi + v
                
                patch_h = y1 - y0
                patch_w = x1 - x0
                
                # Use cv2.getRectSubPix for efficient patch extraction with subpixel accuracy
                g2_patch = cv2.getRectSubPix(
                    gframe2,
                    (patch_w, patch_h),
                    (cx, cy)
                ).astype(np.float64)
                
                diff_patch = g2_patch - g1_patch

                b = np.array([
                    np.dot(Ix_flat, diff_patch.ravel()),
                    np.dot(Iy_flat, diff_patch.ravel()),
                ], dtype=np.float64)

                du, dv = A_inv @ b
                u -= du
                v -= dv

                if abs(du) < epsilon and abs(dv) < epsilon:
                    break

            flows[i] = (u, v)

        return flows


def find_rotation(curr_points,next_points):
    p1,p2 = curr_points
    a = p2-p1
    b = next_points[1]-next_points[0]
    dem = np.norm(a)*np.norm(b)
    en = a@b
    return (en/dem)


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
        
        overlay = cv2.addWeighted(next_frame.astype(np.float32), 0.5, transformed.astype(np.float32), 0.5, 0)
        overlay_path = f"debug_outputs/overlay_frame_{i}_on_{i+1}.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"  Saved {overlay_path}")
    
    print("\nTransform matrices applied and saved to debug_outputs/")
    

frames = read_video_to_frames()
transform_matrices = []

print(f"Processing {len(frames)} frames...")

for idx in range(len(frames) - 1):
    print(f"Frame {idx}/{len(frames)-1}", end='\r')
    g = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(g, maxCorners=4, qualityLevel=0.01, minDistance=60)
    points = corners.reshape(-1, 2)
    flows = find_alignment_between_frames(frames[idx], frames[idx+1], points)
    p1_curr = points[0]
    p1_next = points[0]+flows[0]
    p2_curr = points[1]
    p2_next = points[1]+flows[1]
    valid_flows = flows[~np.isnan(flows).any(axis=1)]
    
    theta = 1*find_rotation((p1_curr,p2_curr),(p1_next,p2_next))
    
    tx = 0
    ty = np.median(valid_flows[:, 1]) 
    
    if idx < 4:
        debug_print_flow(idx, theta, tx, ty, points, flows)
        debug_visualize_flows(frames[idx], points, flows, f"debug_outputs/flow_frame_{idx}.png")
    
    center_x = (p1_curr[0] + p2_curr[0]) / 2
    center_y = (p1_curr[1] + p2_curr[1]) / 2
    
    rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(theta), 1.0)
    rot_matrix[0, 2] += tx
    rot_matrix[1, 2] += ty
    
    M = np.vstack([rot_matrix, [0, 0, 1]])
    transform_matrices.append(M)

print("\nDone processing frames.")

apply_and_visualize_transforms(frames, transform_matrices, num_frames=4)

# Save stabilized video
print("\nCreating stabilized video...")
stabilized_frames = []
h, w = frames[0].shape[:2]

for idx in range(len(frames) - 1):
    print(f"Stabilizing frame {idx}/{len(frames)-1}", end='\r')
    M = transform_matrices[idx]
    stabilized = cv2.warpAffine(frames[idx], M[:2, :], (w, h))
    stabilized_frames.append(stabilized)

# Add last frame without transformation
stabilized_frames.append(frames[-1])

# Save as video
output_path = "debug_outputs/stabilized_video.mp4"
iio.imwrite(output_path, np.array(stabilized_frames), fps=30, plugin="pyav", codec="libx264")
print(f"\nStabilized video saved to {output_path}")

