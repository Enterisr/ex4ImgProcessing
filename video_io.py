"""Video input/output functions."""
import os
import numpy as np
import imageio.v3 as iio


def read_video_to_frames(input_path="input/Garden.mp4"):
    vid = iio.imread(input_path, plugin="pyav")
    
    # Get FPS from video
    props = iio.improps(input_path, plugin="pyav")
    fps = props.fps if hasattr(props, 'fps') else 30
    
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create output directory for this video
    output_dir = f"debug_outputs/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input video: {len(vid)} frames at {fps} FPS ({len(vid)/fps:.2f} seconds)")
    print(f"Output directory: {output_dir}")
    
    return vid, fps, video_name, output_dir


def save_stabilized_video(frames, output_path, fps=30):
    """Save frames as video with specified FPS."""
    iio.imwrite(
        output_path, np.array(frames), fps=fps, plugin="pyav", codec="libx264"
    )
