import subprocess
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def run_compression(quality):
    """Run the C++ compression program with specified quality."""
    try:
        # Run the C++ program with quality parameter
        process = subprocess.Popen(
            ['./main', '-q', str(quality)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Provide the image name when prompted
        output, error = process.communicate(input='img/Boy_1024.png\n')

        if process.returncode != 0:
            print(f"Error running compression with quality {quality}: {error}")
            return False
        return True
    except Exception as e:
        print(f"Exception while running compression with quality {quality}: {e}")
        return False

def create_comparison_image(qualities):
    """Create a single image comparing different quality levels."""
    # Define the grid layout (3x3 for 9 images)
    grid_size = (3, 3)

    # Load first image to get dimensions
    sample_img = cv2.imread(f'output/decompress_image_cpu{qualities[0]}.png')
    if sample_img is None:
        raise Exception("Could not load sample image")

    # Calculate individual image size and padding
    img_h, img_w = sample_img.shape[:2]
    vertical_padding = 40  # Space for text
    horizontal_padding = 20  # New padding between images

    # Create blank canvas with additional width for padding
    canvas_w = (grid_size[1] * img_w) + (horizontal_padding * (grid_size[1] - 1))  # Add padding between images
    canvas_h = grid_size[0] * (img_h + vertical_padding)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Place images and add text
    idx = 0
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            if idx >= len(qualities):
                break

            # Load and place image
            quality = qualities[idx]
            img_path = f'output/decompress_image_cpu{quality}.png'
            img = cv2.imread(img_path)

            if img is not None:
                # Calculate position with horizontal padding
                y_offset = row * (img_h + vertical_padding)
                x_offset = col * (img_w + horizontal_padding)  # Add padding to x offset

                # Place image
                canvas[y_offset+vertical_padding:y_offset+vertical_padding+img_h,
                      x_offset:x_offset+img_w] = img

                # Add text
                text = f"Quality = {quality}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = x_offset + (img_w - text_size[0]) // 2
                text_y = y_offset + vertical_padding - 10

                cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

            idx += 1

    return canvas

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Define quality levels to test (10 to 90 in steps of 10)
    qualities = list(range(10, 91, 10))

    # Run compression for each quality level
    for quality in qualities:
        print(f"Processing quality level: {quality}%")
        success = run_compression(quality)
        if not success:
            print(f"Failed to process quality level {quality}")

    # Create comparison image
    print("Creating comparison image...")
    try:
        comparison = create_comparison_image(qualities)
        output_path = 'output/quality_comparison.png'
        cv2.imwrite(output_path, comparison)
        print(f"Comparison image saved to {output_path}")
    except Exception as e:
        print(f"Error creating comparison image: {e}")

if __name__ == "__main__":
    main()
