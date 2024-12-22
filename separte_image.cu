#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void rgbToYcbcrKernel(unsigned char *d_rgb, unsigned char *d_y, unsigned char *d_cb, unsigned char *d_cr, int width, int height, int channels)
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
        d_y[y * width + x] = Y;
        d_cb[y * width + x] = Cb;
        d_cr[y * width + x] = Cr;
    }
}
void rgbToYcbcr(const cv::Mat &rgbImage, cv::Mat &yImage, cv::Mat &cbImage, cv::Mat &crImage)
{
    int width = rgbImage.cols;
    int height = rgbImage.rows;
    int channels = rgbImage.channels();

    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate device memory
    unsigned char *d_rgb, *d_y, *d_cb, *d_cr;
    cudaMalloc((void **)&d_rgb, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_y, imageSize);
    cudaMalloc((void **)&d_cb, imageSize);
    cudaMalloc((void **)&d_cr, imageSize);

    // Copy image data to device
    cudaMemcpy(d_rgb, rgbImage.data, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configure CUDA grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    rgbToYcbcrKernel<<<gridSize, blockSize>>>(d_rgb, d_y, d_cb, d_cr, width, height, channels);

    // Synchronize and check for errors
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(yImage.data, d_y, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(cbImage.data, d_cb, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(crImage.data, d_cr, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rgb);
    cudaFree(d_y);
    cudaFree(d_cb);
    cudaFree(d_cr);
}

int main()
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

    // Create output images for Y, Cb, and Cr channels
    cv::Mat yImage(rgbImage.size(), CV_8UC1);
    cv::Mat cbImage(rgbImage.size(), CV_8UC1);
    cv::Mat crImage(rgbImage.size(), CV_8UC1);

    // Convert RGB to YCbCr
    rgbToYcbcr(rgbImage, yImage, cbImage, crImage);

    // Save the output images
    cv::imwrite("Y_channel.jpg", yImage);
    cv::imwrite("Cb_channel.jpg", cbImage);
    cv::imwrite("Cr_channel.jpg", crImage);

    std::cout << "Y, Cb, and Cr channels saved as separate images." << std::endl;

    return 0;
}
