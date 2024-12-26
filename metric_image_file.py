import subprocess
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def execute(binary_path, *image_files):
    metrics = []  # To store metrics for each image

    # Ensure binary exists
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Binary file '{binary_path}' not found.")

    widths = []
    heights = []

    # Process each image
    for image_file in image_files:
        if not os.path.exists(image_file):
            print(f"Image file '{image_file}' not found. Skipping...")
            continue

        # Run the binary with the image file

        result = subprocess.run(
            [binary_path, image_file],
            capture_output=True,
            text=True
        )

        print("Captured STDOUT:", result.stdout)  # Debugging step
        print("Captured STDERR:", result.stderr)  # Debugging step

        # Check if there is an error
        if result.returncode != 0:
            raise RuntimeError(f"Error executing binary: {result.stderr}")

        # Split the output and remove empty lines
        output_lines = [
            line for line in result.stdout.splitlines() if line.strip()]

        print(f"Result received: {output_lines}")  # Debugging log
        encoding_gain = float(output_lines[0].split(
            ':')[-1].strip())
        decoding_gain = float(output_lines[1].split(
            ':')[-1].strip())
        compression_ratio_cpu = float(output_lines[2].split(
            ':')[-1].strip())
        compression_ratio_gpu = float(output_lines[3].split(
            ':')[-1].strip())

        # Get file size of the image
        file_size = os.path.getsize(image_file) / 1024  # Size in KB

        # Read image dimensions using Pillow
        with Image.open(image_file) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

        # Append to metrics list
        metrics.append(
            [image_file, file_size, encoding_gain, decoding_gain, compression_ratio_cpu, compression_ratio_gpu])

    if not metrics:
        print("No metrics to plot.")
        return

    # Save metrics to CSV
    csv_file = "result/metrics_image_file.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image File", "File Size (KB)", "Encoding Performance Gain", "Decoding Performance Gain",
                        "Compression Ratio (CPU)", "Compression Ratio (GPU)", "CPU MSE", "CPU PSNR", "GPU MSE", "GPU PSNR"])
        writer.writerows(metrics)

    print(f"Metrics saved to {csv_file}")

    # Extract data for plots
    sizes = [row[1] for row in metrics]
    encoding_gains = [row[2] for row in metrics]
    decoding_gains = [row[3] for row in metrics]
    compression_cpu = [row[4] for row in metrics]
    compression_gpu = [row[5] for row in metrics]

    # First chart: File Size vs Encoding and Decoding Gains, and Dimensions vs Gains (4 subplots)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # File Size vs Encoding Gain
    axs[0, 0].plot(sizes, encoding_gains, color='blue',
                   label='Encoding Gain', marker='o', linestyle='-', markersize=6)
    axs[0, 0].set_title('File Size vs Encoding Gain')
    axs[0, 0].set_xlabel('File Size (KB)')
    axs[0, 0].set_ylabel('Performance Gain (x)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # File Size vs Decoding Gain
    axs[0, 1].plot(sizes, decoding_gains, color='green',
                   label='Decoding Gain', marker='x', linestyle='-', markersize=6)
    axs[0, 1].set_title('File Size vs Decoding Gain')
    axs[0, 1].set_xlabel('File Size (KB)')
    axs[0, 1].set_ylabel('Performance Gain (x)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # File Dimensions vs Encoding Gain
    axs[1, 0].plot(widths, encoding_gains, color='red',
                   label='Encoding Gain', marker='s', linestyle='-', markersize=6)
    axs[1, 0].set_title('File Dimensions vs Encoding Gain')
    axs[1, 0].set_xlabel('Image Dimension (px)')
    axs[1, 0].set_ylabel('Performance Gain (x)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # File Dimensions vs Decoding Gain
    axs[1, 1].plot(widths, decoding_gains, color='purple',
                   label='Decoding Gain', marker='d', linestyle='-', markersize=6)
    axs[1, 1].set_title('File Dimensions vs Decoding Gain')
    axs[1, 1].set_xlabel('Image Dimension (px)')
    axs[1, 1].set_ylabel('Performance Gain (x)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        "result/size_and_dimensions_vs_performance_gains_image_file.png")
    plt.show()

    print("Charts saved as 'result/size_and_dimensions_vs_performance_gains_image_file.png'")

    # Second chart: File Size vs Compression Ratios (side-by-side plots)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(sizes, compression_cpu, color='orange',
                label='Compression Ratio (CPU)', marker='o', linestyle='-', markersize=6)
    axs[0].set_title('File Size vs Compression Ratio (CPU)')
    axs[0].set_xlabel('File Size (KB)')
    axs[0].set_ylabel('Compression Ratio')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(sizes, compression_gpu, color='purple',
                label='Compression Ratio (GPU)', marker='x', linestyle='-', markersize=6)
    axs[1].set_title('File Size vs Compression Ratio (GPU)')
    axs[1].set_xlabel('File Size (KB)')
    axs[1].set_ylabel('Compression Ratio')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        "result/size_vs_compression_ratios_side_by_side_image_file.png")
    plt.show()

    print(
        "Chart saved as result/size_vs_compression_ratios_side_by_side_image_file.png")


# Example usage
execute("./analysis", "./img/01_512_Barbara.png", "./img/Boy_1024.png", )
