#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

using namespace std;

__constant__ float c_dct_table[8][8];

void initDCTTable(float *h_dct_table)
{
    for (int i = 0; i < 8; i++)
    {
        float ci = (i == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
        for (int j = 0; j < 8; j++)
        {
            h_dct_table[i * 8 + j] = ci * cosf((2.0f * j + 1.0f) * i * M_PI / 16.0f);
        }
    }
}

__global__ void recenterKernel(int *input, int num_blocks)
{
    int tx = threadIdx.x;       // Position within the 8x8 block (0-7)
    int ty = threadIdx.y;       // Position within the 8x8 block (0-7)
    int block_idx = blockIdx.x; // Which block we're processing

    if (block_idx < num_blocks)
    {
        // Calculate position in global memory
        int idx = block_idx * 64 + ty * 8 + tx;
        // Subtract 128 from each value
        input[idx] -= 128;
    }
}

__global__ void dctKernel(const int *input, float *output, int num_blocks)
{
    __shared__ float s_block[8][8];
    __shared__ float s_temp[8][8];

    // Thread and block indices
    int tx = threadIdx.x;       // Position within the 8x8 block (0-7)
    int ty = threadIdx.y;       // Position within the 8x8 block (0-7)
    int block_idx = blockIdx.x; // Which image block we're processing

    // Only process if we're within valid blocks
    if (block_idx < num_blocks)
    {
        // Load data into shared memory
        // Each thread loads one pixel from its corresponding image block
        s_block[ty][tx] = input[block_idx * 64 + ty * 8 + tx];
        __syncthreads();

        // Step 1: Row-wise 1D DCT
        float sum = 0.0f;
        for (int j = 0; j < 8; j++)
        {
            sum += s_block[ty][j] * c_dct_table[tx][j];
        }
        s_temp[ty][tx] = sum;
        __syncthreads();

        // Step 2: Column-wise 1D DCT
        sum = 0.0f;
        for (int i = 0; i < 8; i++)
        {
            sum += s_temp[i][tx] * c_dct_table[ty][i];
        }

        // Write result to global memory
        output[block_idx * 64 + ty * 8 + tx] = roundf(sum * 100.0f / 4.0f) / 100.0f;
    }
}

__global__ void quantizeKernel(float *dct_blocks, const int *quantization_table, int num_blocks)
{
    int block_idx = blockIdx.x; // Which block we're processing
    int tx = threadIdx.x;       // Position within the 8x8 block (0-63)

    if (block_idx < num_blocks)
    {
        int idx = block_idx * 64 + tx; // Position in global memory
        dct_blocks[idx] = roundf(dct_blocks[idx] / quantization_table[tx]);
        if (dct_blocks[idx] == -0.0f)
            dct_blocks[idx] = 0.0f; // Handle negative zero
    }
}

void dctQuantizationParallel(const vector<vector<vector<int>>> &input_blocks, const vector<vector<int>> &quantization_table,
                             vector<vector<float>> &quantize_coefficients)
{
    int num_blocks = input_blocks.size();
    int total_size = num_blocks * 64; // Total number of elements

    // Allocate device memory
    int *d_input, *d_block_lengths, *d_quantization_table;
    float *d_output;

    cudaMalloc(&d_input, total_size * sizeof(int));
    cudaMalloc(&d_output, total_size * sizeof(float));
    cudaMalloc(&d_block_lengths, num_blocks * sizeof(int));
    cudaMalloc(&d_quantization_table, 64 * sizeof(int));

    // Flatten quantization table
    vector<int> quantization_flat(64);
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            quantization_flat[i * 8 + j] = quantization_table[i][j];
        }
    }

    // Prepare quantization table
    cudaMemcpy(d_quantization_table, quantization_flat.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);

    // Flatten input blocks for device transfer
    vector<int> input_linear(total_size);
    for (int block = 0; block < num_blocks; block++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                input_linear[block * 64 + i * 8 + j] = input_blocks[block][i][j];
            }
        }
    }
    cudaMemcpy(d_input, input_linear.data(), total_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch recenter kernel
    dim3 blockDim(8, 8);      // Each thread block is 8x8
    dim3 gridDim(num_blocks); // One block per image
    recenterKernel<<<gridDim, blockDim>>>(d_input, num_blocks);

    // Initialize DCT table and copy to device
    float h_dct_table[64];
    initDCTTable(h_dct_table);
    cudaMemcpyToSymbol(c_dct_table, h_dct_table, 64 * sizeof(float));

    // Launch DCT kernel
    dctKernel<<<gridDim, blockDim>>>(d_input, d_output, num_blocks);

    // Launch quantization kernel
    quantizeKernel<<<num_blocks, 64>>>(d_output, d_quantization_table, num_blocks);

    // Copy results back to host
    vector<float> output_linear(total_size);
    cudaMemcpy(output_linear.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Reformat results
    quantize_coefficients.resize(num_blocks);
    for (int b = 0; b < num_blocks; b++)
    {
        quantize_coefficients[b].assign(output_linear.begin() + b * 64, output_linear.begin() + b * 64 + 64);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_lengths);
    cudaFree(d_quantization_table);
}

__global__ void dequantizeKernel(float *dct_blocks, const int *quantization_table, int num_blocks)
{
    int block_idx = blockIdx.x;
    int tx = threadIdx.x;

    if (block_idx < num_blocks)
    {
        int idx = block_idx * 64 + tx;
        dct_blocks[idx] = dct_blocks[idx] * quantization_table[tx];
    }
}

__global__ void idctKernel(float *input, int *output, int num_blocks)
{
    __shared__ float s_block[8][8];
    __shared__ float s_temp[8][8];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_idx = blockIdx.x;

    if (block_idx < num_blocks)
    {
        // Load data into shared memory
        s_block[ty][tx] = input[block_idx * 64 + ty * 8 + tx];
        __syncthreads();

        // Step 1: Row-wise 1D IDCT
        float sum = 0.0f;
        for (int j = 0; j < 8; j++)
        {
            float cj = (j == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
            sum += cj * s_block[ty][j] * cosf((2.0f * tx + 1.0f) * j * M_PI / 16.0f);
        }
        s_temp[ty][tx] = sum;
        __syncthreads();

        // Step 2: Column-wise 1D IDCT
        sum = 0.0f;
        for (int i = 0; i < 8; i++)
        {
            float ci = (i == 0) ? 1.0f / sqrt(2.0f) : 1.0f;
            sum += ci * s_temp[i][tx] * cosf((2.0f * ty + 1.0f) * i * M_PI / 16.0f);
        }

        // Scale, round and add 128 back
        output[block_idx * 64 + ty * 8 + tx] = static_cast<int>(roundf(sum / 4.0f)) + 128;
    }
}

void dequantizeInverseDCTParallel(const vector<vector<vector<float>>> &quantize_coefficients, const vector<vector<int>> &quantization_table,
                                  vector<vector<vector<int>>> &output_blocks)
{
    int num_blocks = quantize_coefficients.size();
    int total_size = num_blocks * 64;

    // Allocate device memory
    int *d_block_lengths, *d_output;
    float *d_dequantized_blocks;
    int *d_quantization_table;

    cudaMalloc(&d_block_lengths, num_blocks * sizeof(int));
    cudaMalloc(&d_dequantized_blocks, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(int));
    cudaMalloc(&d_quantization_table, 64 * sizeof(int));

    // Flatten quantization table
    vector<int> quantization_flat(64);
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            quantization_flat[i * 8 + j] = quantization_table[i][j];
        }
    }

    cudaMemcpy(d_quantization_table, quantization_flat.data(), 64 * sizeof(int),
               cudaMemcpyHostToDevice);

    // Flatten input blocks for device transfer
    vector<float> input_linear(total_size);
    for (int block = 0; block < num_blocks; block++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                input_linear[block * 64 + i * 8 + j] = quantize_coefficients[block][i][j];
            }
        }
    }
    cudaMemcpy(d_dequantized_blocks, input_linear.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch dequantize kernel
    dequantizeKernel<<<num_blocks, 64>>>(d_dequantized_blocks, d_quantization_table, num_blocks);

    dim3 blockDim(8, 8);
    dim3 gridDim(num_blocks);
    idctKernel<<<gridDim, blockDim>>>(d_dequantized_blocks, d_output, num_blocks);

    // Copy results back to host
    vector<int> output_linear(total_size);
    cudaMemcpy(output_linear.data(), d_output, total_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Reshape output into blocks
    output_blocks.resize(num_blocks, vector<vector<int>>(8, vector<int>(8)));
    for (int block = 0; block < num_blocks; block++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                output_blocks[block][i][j] = output_linear[block * 64 + i * 8 + j];
            }
        }
    }

    // Free device memory
    cudaFree(d_block_lengths);
    cudaFree(d_dequantized_blocks);
    cudaFree(d_output);
    cudaFree(d_quantization_table);
}
