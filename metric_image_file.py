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
            text=True,
            timeout=300
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
        encoding_gain_omp = float(output_lines[2].split(
            ':')[-1].strip())
        decoding_gain_omp = float(output_lines[3].split(
            ':')[-1].strip())
        compression_ratio_cpu = float(output_lines[4].split(
            ':')[-1].strip())
        compression_ratio_gpu = float(output_lines[5].split(
            ':')[-1].strip())
        compression_ratio_cpu_omp = float(output_lines[6].split(
            ':')[-1].strip())
        metric_cpu_mse = float(output_lines[7].split(
            ':')[-1].strip())
        metric_cpu_psnr = float(output_lines[8].split(
            ':')[-1].strip())
        metric_gpu_mse = float(output_lines[9].split(
            ':')[-1].strip())
        metric_gpu_psnr = float(output_lines[10].split(
            ':')[-1].strip())
        metric_cpu_mse_omp = float(output_lines[11].split(
            ':')[-1].strip())
        metric_cpu_psnr_omp = float(output_lines[12].split(
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
            [image_file, file_size, encoding_gain, decoding_gain, encoding_gain_omp, decoding_gain_omp,
             compression_ratio_cpu, compression_ratio_gpu, compression_ratio_cpu_omp,
             metric_cpu_mse, metric_cpu_psnr, metric_gpu_mse, metric_gpu_psnr, metric_cpu_mse_omp, metric_cpu_psnr_omp])

    if not metrics:
        print("No metrics to plot.")
        return

    # Save metrics to CSV
    csv_file = "result/metrics_image_file.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image File", "File Size (KB)", "Encoding Performance Gain", "Decoding Performance Gain",
            "Encoding Performance Gain (OMP)", "Decoding Performance Gain (OMP)",
            "Compression Ratio (CPU)", "Compression Ratio (GPU)", "Compression Ratio (CPU OMP)",
            "CPU MSE", "CPU PSNR", "GPU MSE", "GPU PSNR", "CPU MSE (OMP)", "CPU PSNR (OMP)"
        ])
        writer.writerows(metrics)

    print(f"Metrics saved to {csv_file}")

    # Extract data for plots
    sizes = [row[1] for row in metrics]
    encoding_gains = [row[2] for row in metrics]
    decoding_gains = [row[3] for row in metrics]
    encoding_gains_omp = [row[4] for row in metrics]
    decoding_gains_omp = [row[5] for row in metrics]
    compression_cpu = [row[6] for row in metrics]
    compression_gpu = [row[7] for row in metrics]
    compression_cpu_omp = [row[8] for row in metrics]
    metric_cpu_mse = [row[9] for row in metrics]
    metric_cpu_psnr = [row[10] for row in metrics]
    metric_gpu_mse = [row[11] for row in metrics]
    metric_gpu_psnr = [row[12] for row in metrics]
    metric_cpu_mse_omp = [row[13] for row in metrics]
    metric_cpu_psnr_omp = [row[14] for row in metrics]

    # First Plot: File Size vs Encoding/Decoding Gains
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # File Size vs Encoding Gains
    axs[0].plot(sizes, encoding_gains, color='blue',
                label='Encoding Gain', marker='o')
    axs[0].plot(sizes, encoding_gains_omp, color='red',
                label='Encoding Gain OMP', marker='x')
    axs[0].set_title('File Size vs Encoding Gains')
    axs[0].set_xlabel('File Size (KB)')
    axs[0].set_ylabel('Performance Gain (x)')
    axs[0].grid(True)
    axs[0].legend()

    # File Size vs Decoding Gains
    axs[1].plot(sizes, decoding_gains, color='green',
                label='Decoding Gain', marker='o')
    axs[1].plot(sizes, decoding_gains_omp, color='purple',
                label='Decoding Gain OMP', marker='x')
    axs[1].set_title('File Size vs Decoding Gains')
    axs[1].set_xlabel('File Size (KB)')
    axs[1].set_ylabel('Performance Gain (x)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("result/file_size_vs_gains.png")
    plt.show()

    # Second Plot: File Size vs Compression Ratios
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(sizes, compression_cpu, color='blue',
                label='Compression Ratio (CPU)', marker='o')
    axs[0].set_title('File Size vs Compression Ratio (CPU)')
    axs[0].set_xlabel('File Size (KB)')
    axs[0].set_ylabel('Compression Ratio')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(sizes, compression_gpu, color='green',
                label='Compression Ratio (GPU)', marker='o')
    axs[1].set_title('File Size vs Compression Ratio (GPU)')
    axs[1].set_xlabel('File Size (KB)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(sizes, compression_cpu_omp, color='red',
                label='Compression Ratio (OMP)', marker='o')
    axs[2].set_title('File Size vs Compression Ratio (OMP)')
    axs[2].set_xlabel('File Size (KB)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("result/file_size_vs_compression_ratios.png")
    plt.show()

    # Third Plot: File Size vs MSE and PSNR
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # MSE Metrics
    axs[0].plot(sizes, metric_cpu_mse, color='blue',
                label='CPU MSE', marker='o')
    axs[0].plot(sizes, metric_gpu_mse, color='green',
                label='GPU MSE', marker='o')
    axs[0].plot(sizes, metric_cpu_mse_omp, color='red',
                label='OMP MSE', marker='o')
    axs[0].set_title('File Size vs MSE')
    axs[0].set_xlabel('File Size (KB)')
    axs[0].set_ylabel('MSE')
    axs[0].grid(True)
    axs[0].legend()

    # PSNR Metrics
    axs[1].plot(sizes, metric_cpu_psnr, color='blue',
                label='CPU PSNR', marker='o')
    axs[1].plot(sizes, metric_gpu_psnr, color='green',
                label='GPU PSNR', marker='o')
    axs[1].plot(sizes, metric_cpu_psnr_omp, color='red',
                label='OMP PSNR', marker='o')
    axs[1].set_title('File Size vs PSNR')
    axs[1].set_xlabel('File Size (KB)')
    axs[1].set_ylabel('PSNR')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("result/file_size_vs_mse_psnr.png")
    plt.show()


# Example usage
execute("./analysis", "./img/01_512_Barbara.png", './img/Lenna_512.png',
        "./img/Circle_2048.png")
