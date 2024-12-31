from email.mime import image
import subprocess
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_fake_image(width, height):
    """Generate a fake image with validation checks."""
    # Ensure dimensions are multiples of 8
    width = (width // 16) * 16
    height = (height // 16) * 16

    # Generate data with controlled constraints
    data = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):  # For each channel
        data[:, :, i] = np.random.randint(16, 235, (height, width))

    try:
        image = Image.fromarray(data, 'RGB')
        test_data = np.array(image)
        if test_data.shape != (height, width, 3):
            raise ValueError(
                f"Image shape mismatch: {test_data.shape} vs expected {(height, width, 3)}")
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def save_and_verify_image(image, file_path):
    """Save and verify the image file."""
    try:
        image.save(file_path)
        with Image.open(file_path) as verify_img:
            verify_img.verify()
        print(f"Successfully saved and verified: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving/verifying image {file_path}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False


def generate_fake_images(start_size, end_size, jump, output_dir="generated_images"):
    """Generate fake images with error checking."""
    os.makedirs(output_dir, exist_ok=True)
    image_files = []
    current_size = start_size
    i = 0
    while current_size <= end_size:
        current_size = (current_size // 16) * 16
        file_name = f"{i:04}fake_image_{current_size}x{current_size}.png"
        file_path = os.path.join(output_dir, file_name)

        fake_image = generate_fake_image(current_size, current_size)
        if fake_image is None:
            print(f"Skipping size {current_size} due to generation failure.")
            current_size += jump
            continue

        if save_and_verify_image(fake_image, file_path):
            image_files.append(file_path)
        else:
            print(f"Failed to save image of size {current_size}, skipping.")

        current_size += jump
        i += 1

    return image_files


def execute(binary_path, *args):
    """Execute binary with error handling."""

    result = subprocess.run(
        [binary_path] + list(args),
        capture_output=True,
        text=True
    )

    print("Captured STDOUT:", result.stdout)  # Debugging step
    print("Captured STDERR:", result.stderr)  # Debugging step

    # Check if there is an error
    if result.returncode != 0:
        return False, result.stderr

    # Split the output and remove empty lines
    output_lines = [
        line for line in result.stdout.splitlines() if line.strip()]

    return True, output_lines


# Main execution
def main():
    start_size = 512
    end_size = 1024
    jump = 128
    output_dir = "my_generated_images"
    binary_path = "./analysis"

    # delete all files in the output directory
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

    image_files = generate_fake_images(start_size, end_size, jump, output_dir)
    image_files.sort()
    metrics = []

    for i, image_file in enumerate(image_files):
        result_status, result = execute(binary_path, image_file)

        if result_status is True:
            print(f"Result received: {result}")  # Debugging log
            encoding_gain = float(result[0].split(
                ':')[-1].strip())
            decoding_gain = float(result[1].split(
                ':')[-1].strip())
            compression_ratio_cpu = float(result[2].split(
                ':')[-1].strip())
            compression_ratio_gpu = float(result[3].split(
                ':')[-1].strip())
            metric_cpu_mse = float(result[4].split(
                ':')[-1].strip())
            metric_cpu_psnr = float(result[5].split(
                ':')[-1].strip())
            metric_gpu_mse = float(result[6].split(
                ':')[-1].strip())
            metric_gpu_psnr = float(result[7].split(
                ':')[-1].strip())

            # Get file size of the image
            file_size = os.path.getsize(image_file) / 1024  # Size in KB

            # Append to metrics list
            metrics.append(
                [image_file, file_size, encoding_gain, decoding_gain,
                 compression_ratio_cpu, compression_ratio_gpu, metric_cpu_mse, metric_cpu_psnr, metric_gpu_mse, metric_gpu_psnr])

        while result_status is False:
            # Regenerate the image
            print(f"Regenerating image: {image_file}")
            size = int(os.path.basename(
                image_file).split('_')[2].split('x')[0])
            fake_image = generate_fake_image(size, size)
            file_name = f"{i:04}fake_image_{size}x{size}.png"
            file_path = os.path.join(output_dir, file_name)
            fake_images = []
            if save_and_verify_image(fake_image, file_path):
                fake_images.append(file_path)

            result_status, result = execute(binary_path, fake_images[0])

            if result_status is True:
                print(f"Result received: {result}")  # Debugging log
                encoding_gain = float(result[0].split(
                    ':')[-1].strip())
                decoding_gain = float(result[1].split(
                    ':')[-1].strip())
                compression_ratio_cpu = float(result[2].split(
                    ':')[-1].strip())
                compression_ratio_gpu = float(result[3].split(
                    ':')[-1].strip())
                metric_cpu_mse = float(result[4].split(
                    ':')[-1].strip())
                metric_cpu_psnr = float(result[5].split(
                    ':')[-1].strip())
                metric_gpu_mse = float(result[6].split(
                    ':')[-1].strip())
                metric_gpu_psnr = float(result[7].split(
                    ':')[-1].strip())

        # Get file size of the image
        file_size = os.path.getsize(image_file) / 1024  # Size in KB

        # Append to metrics list
        metrics.append(
            [image_file, file_size, encoding_gain, decoding_gain,
             compression_ratio_cpu, compression_ratio_gpu, metric_cpu_mse, metric_cpu_psnr, metric_gpu_mse, metric_gpu_psnr])

    if not metrics:
        print("No metrics to plot.")
        return
    # Save metrics to CSV
    csv_file = "result/metrics_fake_image.csv"

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
    metric_cpu_mse = [row[6] for row in metrics]
    metric_cpu_psnr = [row[7] for row in metrics]
    metric_gpu_mse = [row[8] for row in metrics]
    metric_gpu_psnr = [row[9] for row in metrics]

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
    dimensions = [int(os.path.basename(row[0]).split('_')[2].split('x')[0])
                  for row in metrics]  # Extract dimensions from file names
    axs[1, 0].plot(dimensions, encoding_gains, color='red',
                   label='Encoding Gain', marker='s', linestyle='-', markersize=6)
    axs[1, 0].set_title('File Dimensions vs Encoding Gain')
    axs[1, 0].set_xlabel('Image Dimension (px)')
    axs[1, 0].set_ylabel('Performance Gain (x)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # File Dimensions vs Decoding Gain
    axs[1, 1].plot(dimensions, decoding_gains, color='purple',
                   label='Decoding Gain', marker='d', linestyle='-', markersize=6)
    axs[1, 1].set_title('File Dimensions vs Decoding Gain')
    axs[1, 1].set_xlabel('Image Dimension (px)')
    axs[1, 1].set_ylabel('Performance Gain (x)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("result/size_and_dimensions_vs_performance_gains_fake_image.png")
    plt.show()

    print("Charts saved as 'result/size_and_dimensions_vs_performance_gains_fake_image.png'")

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
    plt.savefig("result/size_vs_compression_ratios_side_by_side.png")
    plt.show()

    print("Chart saved as result/size_vs_compression_ratios_side_by_side_image_file.png")

    # Third chart: MSE and PSNR metrics (2 subplots)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Subplot for MSE metrics (CPU and GPU)
    axs[0].plot(sizes, metric_cpu_mse, color='blue',
                label='CPU MSE', marker='o', linestyle='-', markersize=6)
    axs[0].plot(sizes, metric_gpu_mse, color='red', label='GPU MSE',
                marker='x', linestyle='-', markersize=6)
    axs[0].set_title('File Size vs MSE (CPU and GPU)')
    axs[0].set_xlabel('File Size (KB)')
    axs[0].set_ylabel('Mean Squared Error (MSE)')
    axs[0].grid(True)
    axs[0].legend()

    # Subplot for PSNR metrics (CPU and GPU)
    axs[1].plot(sizes, metric_cpu_psnr, color='green',
                label='CPU PSNR', marker='o', linestyle='-', markersize=6)
    axs[1].plot(sizes, metric_gpu_psnr, color='purple',
                label='GPU PSNR', marker='x', linestyle='-', markersize=6)
    axs[1].set_title('File Size vs PSNR (CPU and GPU)')
    axs[1].set_xlabel('File Size (KB)')
    axs[1].set_ylabel('Peak Signal-to-Noise Ratio (PSNR)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("result/size_vs_mse_psnr_metrics_image_file.png")
    plt.show()

    print("Chart saved as 'result/size_vs_mse_psnr_metrics_image_file.png'")


if __name__ == "__main__":
    main()
