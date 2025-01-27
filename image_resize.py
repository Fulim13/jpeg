import os
from PIL import Image

# Open the image file
image = Image.open("img/lena_color_512.tif")

# Directory to save resized images
output_dir = "img/resized/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Delete all files in the output directory
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    os.remove(file_path)

# Resize and save images with widths from 512 to 4960 in steps of 128
start_width = 512
end_width = 8192
step = 512

for new_width in range(start_width, end_width + 1, step):
    new_height = int((new_width / image.width) * image.height)
    resized_image = image.resize((new_width, new_height))
    # Format width with leading zeros if less than 1000
    formatted_width = f"{new_width:04}" if new_width < 1000 else f"{new_width}"
    output_path = os.path.join(output_dir, f"lena_color_{formatted_width}.tif")
    resized_image.save(output_path)
    print(f"Saved resized image: {output_path}")

print("All images resized and saved successfully.")
