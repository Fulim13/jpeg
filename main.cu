#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel for RGB to YCbCr Conversion
__global__ void rgbToYcbcrKernel(unsigned char *d_rgb, unsigned char *d_ycbcr, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        // Read RGB values
        float R = d_rgb[idx];
        float G = d_rgb[idx + 1];
        float B = d_rgb[idx + 2];

        // Convert to YCbCr
        unsigned char Y = (unsigned char)(0.299 * R + 0.587 * G + 0.114 * B);
        unsigned char Cb = (unsigned char)(-0.168736 * R - 0.331264 * G + 0.5 * B + 128);
        unsigned char Cr = (unsigned char)(0.5 * R - 0.418688 * G - 0.081312 * B + 128);

        // Store results
        d_ycbcr[idx] = Y;
        d_ycbcr[idx + 1] = Cb;
        d_ycbcr[idx + 2] = Cr;
    }
}

void rgbToYcbcr(const cv::Mat &rgbImage, cv::Mat &ycbcrImage)
{
    int width = rgbImage.cols;
    int height = rgbImage.rows;
    int channels = rgbImage.channels();

    size_t imageSize = width * height * channels * sizeof(unsigned char);

    // Allocate device memory
    unsigned char *d_rgb, *d_ycbcr;
    cudaMalloc((void **)&d_rgb, imageSize);
    cudaMalloc((void **)&d_ycbcr, imageSize);

    // Copy image data to device
    cudaMemcpy(d_rgb, rgbImage.data, imageSize, cudaMemcpyHostToDevice);

    // Configure CUDA grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    rgbToYcbcrKernel<<<gridSize, blockSize>>>(d_rgb, d_ycbcr, width, height, channels);

    // Synchronize and check for errors
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(ycbcrImage.data, d_ycbcr, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rgb);
    cudaFree(d_ycbcr);
}

int main2()
{
    // Load image using OpenCV
    cv::Mat rgbImage = cv::imread("Lenna.png");
    if (rgbImage.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Ensure image is in 3-channel format
    cv::cvtColor(rgbImage, rgbImage, cv::COLOR_BGR2RGB);

    // Create output image
    cv::Mat ycbcrImage(rgbImage.size(), rgbImage.type());

    // Convert RGB to YCbCr
    rgbToYcbcr(rgbImage, ycbcrImage);

    // Save the output image
    cv::imwrite("output.jpg", ycbcrImage);

    std::cout << "Image converted to YCbCr and saved as output.jpg" << std::endl;

    return 0;
}
