"""
Generate test video with synthetic motion for debugging video stabilization.
Creates a video with a moving square that has translation and rotation.
"""
import cv2
import numpy as np
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "debug_outputs"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 2.0
VIDEO_CODEC = 'mp4v'

# Square properties
SQUARE_COLOR = (0, 0, 255)  # BGR: Red
BACKGROUND_COLOR = 255  # White
SQUARE_SIZE = 100  # Width and height of square
SQUARE_BASE_X = 100  # Base X position
SQUARE_BASE_Y = 150  # Base Y position

# Frame transformations: (dx, dy, rotation_angle)
# Each tuple defines: (x_translation, y_translation, rotation_in_degrees)
# Translations are cumulative from the base position
FRAME_TRANSFORMS = [
    (100, 100, 0),      # Frame 0: baseline position
    (0, 0, 0),      # Frame 1: move right 1px
    (0, 0, 0),      # Frame 2: move right 1px more
    (0, 0, 2),      # Frame 3: move right 1px more
    (0, 0, 0),      # Frame 4: move right 1px more
    (0, -1, 0),      # Frame 5: move right 1px more
    (0, 0, 0),      # Frame 6: move right 1px more
    (0, -1,0),      # Frame 7: move right 1px more
]

# ============================================================================
# FRAME GENERATION
# ============================================================================

def create_blank_frame(width: int, height: int, color: int = 255) -> np.ndarray:
    """Create a blank white frame."""
    return np.ones((height, width, 3), dtype=np.uint8) * color


def draw_rectangle(frame: np.ndarray, top_left: tuple, bottom_right: tuple, 
                   color: tuple) -> np.ndarray:
    """Draw a filled rectangle on the frame."""
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    return frame


def apply_rotation(frame: np.ndarray, angle: float, center: tuple = None,
                   border_value: tuple = (255, 255, 255)) -> np.ndarray:
    """Apply rotation transformation to frame."""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), borderValue=border_value)


def generate_frame(width: int, height: int, square_size: int,
                  x: float, y: float, rotation: float) -> np.ndarray:
    """
    Generate a frame with square at specified position.
    
    Args:
        width: Frame width
        height: Frame height
        square_size: Size of the square
        x: X position (top-left corner)
        y: Y position (top-left corner)
        rotation: Rotation angle in degrees
        
    Returns:
        Generated frame
    """
    frame = create_blank_frame(width, height)
    
    # Calculate rectangle corners
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + square_size), int(y + square_size)
    
    # Draw rectangle
    frame = draw_rectangle(frame, (x1, y1), (x2, y2), SQUARE_COLOR)
    
    # Apply rotation if needed
    if rotation != 0:
        frame = apply_rotation(frame, rotation)
    
    return frame


# ============================================================================
# VIDEO CREATION
# ============================================================================

def save_frame(frame: np.ndarray, filepath: str, description: str = ""):
    """Save frame as image file."""
    cv2.imwrite(filepath, frame)
    desc_str = f" ({description})" if description else ""
    print(f"Saved {os.path.basename(filepath)}{desc_str}")


def create_video(frames: list, output_path: str, fps: float, 
                frame_size: tuple, codec: str = 'mp4v'):
    """Create video from list of frames."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Saved {os.path.basename(output_path)} with {len(frames)} frames")


def get_transform_description(dx: float, dy: float, rotation: float) -> str:
    """Generate human-readable description of transform."""
    parts = []
    if dx != 0 or dy != 0:
        parts.append(f"Δ({dx:+g}, {dy:+g})")
    if rotation != 0:
        parts.append(f"rot={rotation}°")
    return ", ".join(parts) if parts else "baseline"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate test motion video with synthetic camera shake."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate frames based on transform array
    print(f"Generating {len(FRAME_TRANSFORMS)} test frames...")
    frames = []
    
    # Track cumulative position
    current_x = SQUARE_BASE_X
    current_y = SQUARE_BASE_Y
    
    for idx, (dx, dy, rotation) in enumerate(FRAME_TRANSFORMS):
        # Apply translation
        current_x += dx
        current_y += dy
        
        frame = generate_frame(
            VIDEO_WIDTH, 
            VIDEO_HEIGHT, 
            SQUARE_SIZE,
            current_x, 
            current_y, 
            rotation
        )
        frames.append(frame)
        
        # Save individual frame for debugging
        description = get_transform_description(dx, dy, rotation)
        save_frame(
            frame, 
            f"{OUTPUT_DIR}/square_frame_{idx}.png", 
            description
        )
    
    # Create video
    print("\nCreating video...")
    video_path = f"{OUTPUT_DIR}/square_motion.mp4"
    create_video(
        frames, 
        video_path, 
        VIDEO_FPS, 
        (VIDEO_WIDTH, VIDEO_HEIGHT),
        VIDEO_CODEC
    )
    
    print("\nTest video generation complete!")
    print(f"Total frames: {len(frames)}")


if __name__ == "__main__":
    main()
