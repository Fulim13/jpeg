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
    end_size = 4096
    jump = 64
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
            encoding_gain_omp = float(result[2].split(
                ':')[-1].strip())
            decoding_gain_omp = float(result[3].split(
                ':')[-1].strip())
            compression_ratio_cpu = float(result[4].split(
                ':')[-1].strip())
            compression_ratio_gpu = float(result[5].split(
                ':')[-1].strip())
            compression_ratio_cpu_omp = float(result[6].split(
                ':')[-1].strip())
            metric_cpu_mse = float(result[7].split(
                ':')[-1].strip())
            metric_cpu_psnr = float(result[8].split(
                ':')[-1].strip())
            metric_gpu_mse = float(result[9].split(
                ':')[-1].strip())
            metric_gpu_psnr = float(result[10].split(
                ':')[-1].strip())
            metric_cpu_mse_omp = float(result[11].split(
                ':')[-1].strip())
            metric_cpu_psnr_omp = float(result[12].split(
                ':')[-1].strip())

            # Get file size of the image
            file_size = os.path.getsize(image_file) / 1024  # Size in KB

            # Append to metrics list
            metrics.append(
                [image_file, file_size, encoding_gain, decoding_gain, encoding_gain_omp, decoding_gain_omp,
                 compression_ratio_cpu, compression_ratio_gpu, compression_ratio_cpu_omp,
                 metric_cpu_mse, metric_cpu_psnr, metric_gpu_mse, metric_gpu_psnr, metric_cpu_mse_omp, metric_cpu_psnr_omp])

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
                    [image_file, file_size, encoding_gain, decoding_gain, encoding_gain_omp, decoding_gain_omp,
                     compression_ratio_cpu, compression_ratio_gpu, compression_ratio_cpu_omp,
                     metric_cpu_mse, metric_cpu_psnr, metric_gpu_mse, metric_gpu_psnr, metric_cpu_mse_omp, metric_cpu_psnr_omp])

    if not metrics:
        print("No metrics to plot.")
        return
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
                label='Encoding Gain GPU', marker='o')
    axs[0].plot(sizes, encoding_gains_omp, color='red',
                label='Encoding Gain OMP (2 threads)', marker='x')
    axs[0].set_title('File Size vs Encoding Gains')
    axs[0].set_xlabel('File Size (KB)')
    axs[0].set_ylabel('Performance Gain (x)')
    axs[0].grid(True)
    axs[0].legend()

    # File Size vs Decoding Gains
    axs[1].plot(sizes, decoding_gains, color='green',
                label='Decoding Gain GPU', marker='o')
    axs[1].plot(sizes, decoding_gains_omp, color='purple',
                label='Decoding Gain OMP (2 threads)', marker='x')
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


if __name__ == "__main__":
    main()
